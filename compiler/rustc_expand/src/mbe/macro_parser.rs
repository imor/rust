//! This is an NFA-based parser, which calls out to the main Rust parser for named non-terminals
//! (which it commits to fully when it hits one in a grammar). There's a set of current NFA threads
//! and a set of next ones. Instead of NTs, we have a special case for Kleene star. The big-O, in
//! pathological cases, is worse than traditional use of NFA or Earley parsing, but it's an easier
//! fit for Macro-by-Example-style rules.
//!
//! (In order to prevent the pathological case, we'd need to lazily construct the resulting
//! `NamedMatch`es at the very end. It'd be a pain, and require more memory to keep around old
//! matcher positions, but it would also save overhead)
//!
//! We don't say this parser uses the Earley algorithm, because it's unnecessarily inaccurate.
//! The macro parser restricts itself to the features of finite state automata. Earley parsers
//! can be described as an extension of NFAs with completion rules, prediction rules, and recursion.
//!
//! Quick intro to how the parser works:
//!
//! A "matcher position" (a.k.a. "position" or "mp") is a dot in the middle of a matcher, usually
//! written as a `·`. For example `· a $( a )* a b` is one, as is `a $( · a )* a b`.
//!
//! The parser walks through the input a token at a time, maintaining a list
//! of threads consistent with the current position in the input string: `cur_mps`.
//!
//! As it processes them, it fills up `eof_mps` with threads that would be valid if
//! the macro invocation is now over, `bb_mps` with threads that are waiting on
//! a Rust non-terminal like `$e:expr`, and `next_mps` with threads that are waiting
//! on a particular token. Most of the logic concerns moving the · through the
//! repetitions indicated by Kleene stars. The rules for moving the · without
//! consuming any input are called epsilon transitions. It only advances or calls
//! out to the real Rust parser when no `cur_mps` threads remain.
//!
//! Example:
//!
//! ```text, ignore
//! Start parsing a a a a b against [· a $( a )* a b].
//!
//! Remaining input: a a a a b
//! next: [· a $( a )* a b]
//!
//! - - - Advance over an a. - - -
//!
//! Remaining input: a a a b
//! cur: [a · $( a )* a b]
//! Descend/Skip (first position).
//! next: [a $( · a )* a b]  [a $( a )* · a b].
//!
//! - - - Advance over an a. - - -
//!
//! Remaining input: a a b
//! cur: [a $( a · )* a b]  [a $( a )* a · b]
//! Follow epsilon transition: Finish/Repeat (first position)
//! next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]
//!
//! - - - Advance over an a. - - - (this looks exactly like the last step)
//!
//! Remaining input: a b
//! cur: [a $( a · )* a b]  [a $( a )* a · b]
//! Follow epsilon transition: Finish/Repeat (first position)
//! next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]
//!
//! - - - Advance over an a. - - - (this looks exactly like the last step)
//!
//! Remaining input: b
//! cur: [a $( a · )* a b]  [a $( a )* a · b]
//! Follow epsilon transition: Finish/Repeat (first position)
//! next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]
//!
//! - - - Advance over a b. - - -
//!
//! Remaining input: ''
//! eof: [a $( a )* a b ·]
//! ```

pub(crate) use MetaVarMatch::*;
pub(crate) use ParseResult::*;

use crate::mbe::{macro_rules::Tracker, KleeneOp, TokenTree};

use rustc_ast::token::{self, DocComment, Nonterminal, NonterminalKind, Token};
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lrc;
use rustc_errors::ErrorGuaranteed;
use rustc_parse::parser::{NtOrTt, Parser};
use rustc_span::symbol::Ident;
use rustc_span::symbol::MacroRulesNormalizedIdent;
use rustc_span::Span;
use std::borrow::Cow;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::fmt::Display;
use std::rc::Rc;

/// A unit within a matcher that a `MatcherPos` can refer to. Similar to (and derived from)
/// `mbe::TokenTree`, but designed specifically for fast and easy traversal during matching.
/// Notable differences to `mbe::TokenTree`:
/// - It is non-recursive, i.e. there is no nesting.
/// - The end pieces of each sequence (the separator, if present, and the Kleene op) are
///   represented explicitly, as is the very end of the matcher.
///
/// This means a matcher can be represented by `&[MatcherLoc]`, and traversal mostly involves
/// simply incrementing the current matcher position index by one.
#[derive(Debug, PartialEq, Clone)]
pub(crate) enum MatcherLoc {
    /// A raw token like abc, ; or =>
    Token { token: Token },
    /// A delimited sequence like (abc)
    Delimited,
    /// A repetition sequence
    Sequence {
        op: KleeneOp,
        num_metavar_decls: usize,
        idx_first_after: usize,
        next_metavar: usize,
        seq_depth: usize,
    },
    /// A kleene operator after a repetition sequence without a separator before it
    SequenceKleeneOpNoSep { op: KleeneOp, idx_first: usize },
    /// A separator after a repetition sequence
    SequenceSep { separator: Token },
    /// A kleene operator after a repetition sequence with a separator before it
    SequenceKleeneOpAfterSep { idx_first: usize },
    /// A meta-variable declaration. For example $e:expr
    MetaVarDecl {
        span: Span,
        bind: Ident,
        kind: Option<NonterminalKind>,
        next_metavar: usize,
        seq_depth: usize,
    },
    /// Last location after every other MatcherLoc
    Eof,
}

impl MatcherLoc {
    pub(super) fn span(&self) -> Option<Span> {
        match self {
            MatcherLoc::Token { token } => Some(token.span),
            MatcherLoc::Delimited => None,
            MatcherLoc::Sequence { .. } => None,
            MatcherLoc::SequenceKleeneOpNoSep { .. } => None,
            MatcherLoc::SequenceSep { .. } => None,
            MatcherLoc::SequenceKleeneOpAfterSep { .. } => None,
            MatcherLoc::MetaVarDecl { span, .. } => Some(*span),
            MatcherLoc::Eof => None,
        }
    }
}

impl Display for MatcherLoc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatcherLoc::Token { token } | MatcherLoc::SequenceSep { separator: token } => {
                write!(f, "`{}`", pprust::token_to_string(token))
            }
            MatcherLoc::MetaVarDecl { bind, kind, .. } => {
                write!(f, "meta-variable `${bind}")?;
                if let Some(kind) = kind {
                    write!(f, ":{}", kind)?;
                }
                write!(f, "`")?;
                Ok(())
            }
            MatcherLoc::Eof => f.write_str("end of macro"),

            // These are not printed in the diagnostic
            MatcherLoc::Delimited => f.write_str("delimiter"),
            MatcherLoc::Sequence { .. } => f.write_str("sequence start"),
            MatcherLoc::SequenceKleeneOpNoSep { op, .. } => write!(f, "sequence operator {op}"),
            MatcherLoc::SequenceKleeneOpAfterSep { .. } => write!(f, "sequence operator"),
        }
    }
}

/// Converts a list of token trees into a list of matcher locations. This is called only for LHS.
pub(super) fn compute_locs(matcher: &[TokenTree]) -> Vec<MatcherLoc> {
    fn inner(
        tts: &[TokenTree],
        locs: &mut Vec<MatcherLoc>,
        next_metavar: &mut usize,
        seq_depth: usize,
    ) {
        debug!("Inside compute_locs::inner, input token trees: {tts:?}");
        for tt in tts {
            match tt {
                TokenTree::Token(token) => {
                    locs.push(MatcherLoc::Token { token: token.clone() });
                }
                TokenTree::Delimited(span, delimited) => {
                    let open_token = Token::new(token::OpenDelim(delimited.delim), span.open);
                    let close_token = Token::new(token::CloseDelim(delimited.delim), span.close);

                    locs.push(MatcherLoc::Delimited);
                    locs.push(MatcherLoc::Token { token: open_token });
                    let tts = &delimited.tts;
                    debug!(
                        "Inside compute_locs::TokenTree::Delimited calling inner with tts {tts:?}"
                    );
                    inner(&delimited.tts, locs, next_metavar, seq_depth);
                    locs.push(MatcherLoc::Token { token: close_token });
                }
                TokenTree::Sequence(_, seq) => {
                    // We can't determine `idx_first_after` and construct the final
                    // `MatcherLoc::Sequence` until after `inner()` is called and the sequence end
                    // pieces are processed. So we push a dummy value (`Eof` is cheapest to
                    // construct) now, and overwrite it with the proper value below.
                    let dummy = MatcherLoc::Eof;
                    locs.push(dummy);

                    let next_metavar_orig = *next_metavar;
                    let op = seq.kleene.op;
                    let idx_first = locs.len();
                    let idx_seq = idx_first - 1;
                    let tts = &seq.tts;
                    debug!(
                        "Inside compute_locs::TokenTree::Sequence calling inner with tts {tts:?}"
                    );
                    inner(&seq.tts, locs, next_metavar, seq_depth + 1);

                    if let Some(separator) = &seq.separator {
                        locs.push(MatcherLoc::SequenceSep { separator: separator.clone() });
                        locs.push(MatcherLoc::SequenceKleeneOpAfterSep { idx_first });
                    } else {
                        locs.push(MatcherLoc::SequenceKleeneOpNoSep { op, idx_first });
                    }

                    // Overwrite the dummy value pushed above with the proper value.
                    locs[idx_seq] = MatcherLoc::Sequence {
                        op,
                        num_metavar_decls: seq.num_captures,
                        idx_first_after: locs.len(),
                        next_metavar: next_metavar_orig,
                        seq_depth,
                    };
                }
                &TokenTree::MetaVarDecl(span, bind, kind) => {
                    locs.push(MatcherLoc::MetaVarDecl {
                        span,
                        bind,
                        kind,
                        next_metavar: *next_metavar,
                        seq_depth,
                    });
                    *next_metavar += 1;
                }
                TokenTree::MetaVar(..) | TokenTree::MetaVarExpr(..) => unreachable!(),
            }
        }
        debug!("Inside compute_locs::inner, output matcher locations: {locs:?}");
    }

    let mut locs = vec![];
    let mut next_metavar = 0;
    inner(matcher, &mut locs, &mut next_metavar, /* seq_depth */ 0);

    // A final entry is needed for eof.
    locs.push(MatcherLoc::Eof);

    debug!("Inside compute_locs, output matcher locations: {locs:?}");
    locs
}

/// A cursor through the matcher, representing the state of matching.
#[derive(Debug)]
struct MatchCursor {
    /// The index into `TtParser::locs`, which represents the "dot".
    idx: usize,

    /// The matches made against metavar decls so far. On a successful match, this vector ends up
    /// with one element per metavar decl in the matcher. Each element records token trees matched
    /// against the relevant metavar by the black box parser. An element will be a `MatchedSeq` if
    /// the corresponding metavar decl is within a sequence.
    ///
    /// It is critical to performance that this is an `Rc`, because it gets cloned frequently when
    /// processing sequences. Mostly for sequence-ending possibilities that must be tried but end
    /// up failing.
    matches: Rc<Vec<MetaVarMatch>>,
}

// This type is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(MatchCursor, 16);

impl MatchCursor {
    /// Adds `m` as a named match for the `metavar_idx`-th metavar. There are only two call sites,
    /// and both are hot enough to be always worth inlining.
    #[inline(always)]
    fn push_match(&mut self, metavar_idx: usize, seq_depth: usize, m: MetaVarMatch) {
        let matches = Rc::make_mut(&mut self.matches);
        match seq_depth {
            0 => {
                // We are not within a sequence. Just append `m`.
                assert_eq!(metavar_idx, matches.len());
                matches.push(m);
            }
            _ => {
                // We are within a sequence. Find the final `MatchedSeq` at the appropriate depth
                // and append `m` to its vector.
                let mut curr = &mut matches[metavar_idx];
                for _ in 0..seq_depth - 1 {
                    match curr {
                        MatchedSeq(seq) => curr = seq.last_mut().unwrap(),
                        _ => unreachable!(),
                    }
                }
                match curr {
                    MatchedSeq(seq) => seq.push(m),
                    _ => unreachable!(),
                }
            }
        }
    }
}

#[derive(Debug)]
enum EofMatchCursors {
    None,
    One(MatchCursor),
    Multiple,
}

/// Represents the possible results of an attempted parse.
pub(crate) enum ParseResult<T, F> {
    /// Parsed successfully.
    ArmMatchSucceeded(T),
    /// Created when an arm fails to match. There are two ways this can happen: either there
    /// were more matcher locations than the input token tree or there were more tokens in
    /// the token tree than the the matcher locations. The first case indicates the invocation
    /// ended unexpectedly and the second indicates that no match arms expected the extra tokens.
    RetryNextArmBecauseArmMatchFailed(F),
    /// Fatal error (malformed macro?). Abort compilation.
    AbortBecauseFatalError(rustc_span::Span, String),
    AbortBecauseErrorAlreadyReported(ErrorGuaranteed),
}

/// A `ParseResult` where the `Success` variant contains a mapping of
/// `MacroRulesNormalizedIdent`s to `NamedMatch`es. This represents the mapping
/// of metavars to the token trees they bind to.
pub(crate) type NamedParseResult<F> = ParseResult<NamedMatches, F>;

/// Contains a mapping of `MacroRulesNormalizedIdent`s to `NamedMatch`es.
/// This represents the mapping of metavars to the token trees they bind to.
pub(crate) type NamedMatches = FxHashMap<MacroRulesNormalizedIdent, MetaVarMatch>;

/// Count how many metavars declarations are in `matcher`.
pub(super) fn count_metavar_decls(matcher: &[TokenTree]) -> usize {
    matcher
        .iter()
        .map(|tt| match tt {
            TokenTree::MetaVarDecl(..) => 1,
            TokenTree::Sequence(_, seq) => seq.num_captures,
            TokenTree::Delimited(_, delim) => count_metavar_decls(&delim.tts),
            TokenTree::Token(..) => 0,
            TokenTree::MetaVar(..) | TokenTree::MetaVarExpr(..) => unreachable!(),
        })
        .sum()
}

/// `MetaVarMatch` is a pattern-match result for a single metavar. All
/// `MatchedNonterminal`s in the `NamedMatch` have the same non-terminal type
/// (expr, item, etc).
///
/// The in-memory structure of a particular `NamedMatch` represents the match
/// that occurred when a particular subset of a matcher was applied to a
/// particular token tree.
///
/// The width of each `MatchedSeq` in the `NamedMatch`, and the identity of
/// the `MatchedNtNonTts`s, will depend on the token tree it was applied
/// to: each `MatchedSeq` corresponds to a single repetition in the originating
/// token tree. The depth of the `NamedMatch` structure will therefore depend
/// only on the nesting depth of repetitions in the originating token tree it
/// was derived from.
///
/// In layperson's terms: `NamedMatch` will form a tree representing nested matches of a particular
/// meta variable. For example, if we are matching the following macro against the following
/// invocation...
///
/// ```rust
/// macro_rules! foo {
///   ($($($x:ident),+);+) => {}
/// }
///
/// foo!(a, b, c, d; a, b, c, d, e);
/// ```
///
/// Then, the tree will have the following shape:
///
/// ```ignore (private-internal)
/// # use NamedMatch::*;
/// MatchedSeq([
///   MatchedSeq([
///     MatchedNonterminal(a),
///     MatchedNonterminal(b),
///     MatchedNonterminal(c),
///     MatchedNonterminal(d),
///   ]),
///   MatchedSeq([
///     MatchedNonterminal(a),
///     MatchedNonterminal(b),
///     MatchedNonterminal(c),
///     MatchedNonterminal(d),
///     MatchedNonterminal(e),
///   ])
/// ])
/// ```
#[derive(Debug, Clone)]
pub(crate) enum MetaVarMatch {
    MatchedSeq(Vec<MetaVarMatch>),

    // A metavar match of type `tt`.
    MatchedTokenTree(rustc_ast::tokenstream::TokenTree),

    // A metavar match of any type other than `tt`.
    MatchedNonterminal(Lrc<Nonterminal>),
}

/// Performs a token equality check, ignoring syntax context (that is, an unhygienic comparison)
fn token_name_eq(t1: &Token, t2: &Token) -> bool {
    if let (Some((ident1, is_raw1)), Some((ident2, is_raw2))) = (t1.ident(), t2.ident()) {
        ident1.name == ident2.name && is_raw1 == is_raw2
    } else if let (Some(ident1), Some(ident2)) = (t1.lifetime(), t2.lifetime()) {
        ident1.name == ident2.name
    } else {
        t1.kind == t2.kind
    }
}

// Note: the vectors could be created and dropped within `parse_tt`, but to avoid excess
// allocations we have a single vector for each kind that is cleared and reused repeatedly.
pub struct TtParser {
    macro_name: Ident,

    /// The set of current matcher positions to be processed. This should be empty by the end of a successful
    /// execution of `match_token`.
    cur_match_cursors: Vec<MatchCursor>,

    /// The set of newly generated matcher positions. These are used to replenish `cur_matcher_positions` in the function
    /// `parse_tt`.
    next_match_cursors: Vec<MatchCursor>,

    /// The set of mps that are waiting for the black-box parser.
    black_box_match_cursors: Vec<MatchCursor>,

    /// Pre-allocate an empty match array, so it can be cloned cheaply for macros with many rules
    /// that have no metavars.
    empty_matches: Rc<Vec<MetaVarMatch>>,
}

impl TtParser {
    pub(super) fn new(macro_name: Ident) -> TtParser {
        TtParser {
            macro_name,
            cur_match_cursors: vec![],
            next_match_cursors: vec![],
            black_box_match_cursors: vec![],
            empty_matches: Rc::new(vec![]),
        }
    }

    pub(super) fn has_no_remaining_items_for_step(&self) -> bool {
        self.cur_match_cursors.is_empty()
    }

    /// Tries to match a token from the invocation's token tree with the matcher.
    /// If the token matches a matcher loc then a match cursor one past the matched
    /// index is added in the `next_match_cursors`.
    ///
    /// Process the match cursors of `cur_match_cursors` until it is empty. In the process,
    /// this will produce more match cursors in `next_match_cursors` and
    /// `black_box_match_cursors`.
    ///
    /// Arguments:
    /// - `matcher` is the matcher it tries to match the token against.
    /// - `token` is the token it tries to match with the matcher
    ///
    /// # Returns
    ///
    /// `Some(result)` if everything is finished, `None` otherwise. Note that matches are kept
    /// track of through the match cursors generated.
    fn match_token<'matcher, T: Tracker<'matcher>>(
        &mut self,
        matcher: &'matcher [MatcherLoc],
        token: &Token,
        approx_position: usize,
        track: &mut T,
    ) -> Option<NamedParseResult<T::Failure>> {
        debug!("Inside match_token. token: {token:?}");
        // Match cursors that would be valid if the macro invocation was over now. Only
        // modified if `token == Eof`.
        let mut eof_match_cursors = EofMatchCursors::None;

        while let Some(mut match_cursor) = self.cur_match_cursors.pop() {
            let matcher_loc = &matcher[match_cursor.idx];
            debug!("current match matcher_loc: {matcher_loc:?}");
            track.before_match_loc(self, matcher_loc);

            match matcher_loc {
                MatcherLoc::Token { token: t } => {
                    // If it's a doc comment, we just ignore it and move on to the next matcher
                    // loc in the matcher. This is a bug, but #95267 showed that existing
                    // programs rely on this behaviour, and changing it would require some
                    // care and a transition period.
                    //
                    // If the token matches, we can just advance the parser.
                    //
                    // Otherwise, this match has failed, there is nothing to do, and hopefully
                    // another match cursor in `cur_match_cursors` will match.
                    if matches!(t, Token { kind: DocComment(..), .. }) {
                        match_cursor.idx += 1;
                        debug!("match_token: MatcherLoc::Token skipping doc comment token.");
                        self.cur_match_cursors.push(match_cursor);
                    } else if token_name_eq(&t, token) {
                        match_cursor.idx += 1;
                        debug!("match_token: MatcherLoc::Token token {token:?} matched.");
                        self.next_match_cursors.push(match_cursor);
                    }
                }
                MatcherLoc::Delimited => {
                    // A delimited is always followed by an OpenDelim (See `compute_locs` function).
                    // Hence it can be just skipped. It looks like Delimited is useless.
                    // If it was removed entirely everything would still work.
                    match_cursor.idx += 1;
                    debug!("match_token: MatcherLoc::Delimited skipping delimited.");
                    self.cur_match_cursors.push(match_cursor);
                }
                &MatcherLoc::Sequence {
                    op,
                    num_metavar_decls,
                    idx_first_after,
                    next_metavar,
                    seq_depth,
                } => {
                    // Matching a sequence means we may have to emit two match cursors into the
                    // `cur_match_cursors`. First by skipping the sequence entirely if it is
                    // possible for the sequence to repeat zero times. And the second by entering
                    // the sequence.

                    // Install an empty vec for each meta-variable within the sequence. This is
                    // in anticipation of the meta-variable matches that we will find inside the
                    // sequence.
                    for metavar_idx in next_metavar..next_metavar + num_metavar_decls {
                        match_cursor.push_match(metavar_idx, seq_depth, MatchedSeq(vec![]));
                    }

                    // If the sequence can have zero matches we try zero matches by skipping
                    // over this sequence.
                    // if matches!(op, KleeneOp::ZeroOrMore | KleeneOp::ZeroOrOne) {
                    if op.can_repeat_zero_times() {
                        // Create a match cursor which points at the match loc just after the
                        // sequence
                        let match_cursor = MatchCursor {
                            idx: idx_first_after,
                            matches: Rc::clone(&match_cursor.matches),
                        };
                        let matcher_loc = &matcher[idx_first_after];
                        debug!(
                            "match_token: MatcherLoc::Sequence skipping `*` or `?` sequence. New matcher_loc: {matcher_loc:?}."
                        );
                        self.cur_match_cursors.push(match_cursor);
                    }

                    // Try one or more matches of this sequence, by entering it.
                    // Skip the (current) MatcherLoc::Sequence and end up at either
                    // a MatcherLoc::SequenceSep or at MatcherLoc::SequenceKleeneOpNoSep one of
                    // which will be the next matcher location.
                    match_cursor.idx += 1;
                    let matcher_loc = &matcher[match_cursor.idx];
                    debug!(
                        "match_token: MatcherLoc::Sequence entering sequence. New matcher_loc: {matcher_loc}."
                    );
                    debug!("match_token: MatcherLoc::Sequence cur_mps.push({match_cursor:?}).");
                    self.cur_match_cursors.push(match_cursor);
                }
                &MatcherLoc::SequenceKleeneOpNoSep { op, idx_first } => {
                    // We are past the end of a sequence with no separator. We might have to emit
                    // two match cursors into the `cur_match_cursors`. One which ends this sequence.
                    // And another to retry the sequence, but only if this sequence can repeat
                    // more than once.
                    let ending_match_cursor = MatchCursor {
                        idx: match_cursor.idx + 1, // +1 skips the Kleene op
                        matches: Rc::clone(&match_cursor.matches),
                    };
                    let matcher_loc = &matcher[ending_match_cursor.idx];
                    debug!(
                        "match_token: MatcherLoc::SequenceKleeneOpNoSep end a sequence no sep. New matcher_loc: {matcher_loc:?}"
                    );
                    self.cur_match_cursors.push(ending_match_cursor);

                    // Try another repetition of the sequence but only if it can repeat more than
                    // once. We can't use it if the repetition is ZeroOrOne because we have just
                    // seen one repetition.
                    if op.can_repeat_more_than_once() {
                        match_cursor.idx = idx_first;
                        let matcher_loc = &matcher[idx_first];
                        debug!(
                            "match_token: MatcherLoc::SequenceKleeneOpNoSep repeat a sequence with sep. New matcher_loc: {matcher_loc:?}"
                        );
                        self.cur_match_cursors.push(match_cursor);
                    }
                }
                MatcherLoc::SequenceSep { separator } => {
                    // We are at the end of a sequence with a separator. We might have to emit two
                    // match cursors. One into `cur_match_cursors` that is past the separator. This
                    // is an attempt at ending the sequence. The second is into `next_match_cursors`
                    // if the current token matches the separator exactly.
                    let ending_match_cursor = MatchCursor {
                        idx: match_cursor.idx + 2, // +2 skips the separator and the Kleene op
                        matches: Rc::clone(&match_cursor.matches),
                    };
                    let matcher_loc = &matcher[ending_match_cursor.idx];
                    debug!(
                        "match_token: MatcherLoc::SequenceSep end a sequence with a sep. New matcher_loc: {matcher_loc:?}"
                    );
                    self.cur_match_cursors.push(ending_match_cursor);

                    if token_name_eq(token, separator) {
                        // The separator matches the current token. Advance past it. We will end
                        // up at MatcherLoc::SequenceKleeneOpAfterSep
                        match_cursor.idx += 1;
                        let matcher_loc = &matcher[match_cursor.idx];
                        debug!(
                            "match_token: MatcherLoc::SequenceSep token matched separator. New matcher_loc: {matcher_loc:?}"
                        );
                        self.next_match_cursors.push(match_cursor);
                    }
                }
                &MatcherLoc::SequenceKleeneOpAfterSep { idx_first } => {
                    // We are at a Kleene operator after a separator. This can't be a ZeroOrOne
                    // Kleene op because it doesn't allow a separator. Here we just try
                    // another repetition. We don't try to end the sequence here because that was
                    // already done while handling MatchLoc::SequenceSep.

                    match_cursor.idx = idx_first;
                    let matcher_loc = &matcher[idx_first];
                    debug!(
                        "match_token: MatcherLoc::SequenceKleeneOpAfterSep repeat a sequence with a sep. New matcher_loc: {matcher_loc:?}"
                    );
                    self.cur_match_cursors.push(match_cursor);
                }
                &MatcherLoc::MetaVarDecl { span, kind, .. } => {
                    // We are in a meta-variable declaration. We can't find matches for a
                    // meta-variable ourselves. We need to add the current match cursor to
                    // `black_box_match_cursors` so that our caller (`parse_tt`) can
                    // invoke the black box parser to get a match for the meta-variable.

                    // Built-in non-terminals never start with these tokens, so we can eliminate
                    // them from consideration. We use the span of the meta-variable declaration
                    // to determine any edition-specific matching behavior for non-terminals.
                    if let Some(kind) = kind {
                        if Parser::nonterminal_may_begin_with(kind, token) {
                            debug!(
                                "match_token: MatcherLoc::MetaVarDecl token can start meta-variable of type {kind}. Adding to bb_mps."
                            );
                            self.black_box_match_cursors.push(match_cursor);
                        } else {
                            debug!(
                                "match_token: MatcherLoc::MetaVarDecl token can't start meta-variable of type {kind}"
                            );
                        }
                    } else {
                        // E.g. `$e` instead of `$e:expr`, reported as a hard error if actually used.
                        // Both this check and the one in `nameize` are necessary, surprisingly.
                        debug!(
                            "match_token: MatcherLoc::MetaVarDecl error: missing fragment specifier"
                        );
                        return Some(AbortBecauseFatalError(
                            span,
                            "missing fragment specifier".to_string(),
                        ));
                    }
                }
                MatcherLoc::Eof => {
                    // We are past the matcher's end, and not in a sequence. Try to end things.
                    debug_assert_eq!(match_cursor.idx, matcher.len() - 1);
                    if *token == token::Eof {
                        eof_match_cursors = match eof_match_cursors {
                            EofMatchCursors::None => EofMatchCursors::One(match_cursor),
                            EofMatchCursors::One(_) | EofMatchCursors::Multiple => {
                                EofMatchCursors::Multiple
                            }
                        };
                        debug!("match_token: eof_mps: {eof_match_cursors:?}.");
                    } else {
                        debug!("match_token: matcher loc is eof but token is not");
                    }
                }
            }
        }

        // If we reached the end of input, check that there is EXACTLY ONE possible matcher.
        // Otherwise, either the parse is ambiguous (which is an error) or there is a syntax error.
        if *token == token::Eof {
            Some(match eof_match_cursors {
                EofMatchCursors::One(mut eof_mp) => {
                    // Need to take ownership of the matches from within the `Rc`.
                    Rc::make_mut(&mut eof_mp.matches);
                    let matches = Rc::try_unwrap(eof_mp.matches).unwrap().into_iter();
                    self.ensure_unique_metavariables(matcher, matches)
                }
                EofMatchCursors::Multiple => {
                    debug!(
                        "match_token: MatcherLoc::MetaVarDecl error: ambiguity: multiple successful parses"
                    );
                    AbortBecauseFatalError(
                        token.span,
                        "ambiguity: multiple successful parses".to_string(),
                    )
                }
                EofMatchCursors::None => {
                    debug!(
                        "match_token: MatcherLoc::MetaVarDecl error: missing tokens in macro arguments"
                    );
                    RetryNextArmBecauseArmMatchFailed(T::build_failure(
                        Token::new(
                            token::Eof,
                            if token.span.is_dummy() {
                                token.span
                            } else {
                                token.span.shrink_to_hi()
                            },
                        ),
                        approx_position,
                        "missing tokens in macro arguments",
                    ))
                }
            })
        } else {
            None
        }
    }

    /// Match the token stream from `parser` against `matcher`.
    /// Arguments:
    /// - `parser` is the main rust grammar parser which helps in parsing non-terminals like expr, stmt etc.
    /// - `matcher` is the matcher we are trying to match against
    pub(super) fn parse_tt<'matcher, T: Tracker<'matcher>>(
        &mut self,
        parser: &mut Cow<'_, Parser<'_>>,
        matcher: &'matcher [MatcherLoc],
        track: &mut T,
    ) -> NamedParseResult<T::Failure> {
        debug!("Inside parse_tt. Matcher: {matcher:?}");
        // A queue of possible match cursors. We initialize it with the match cursor in
        // which the "dot" is before the first token of the first token tree in `matcher`.
        // `match_token` then processes all of these possible match cursors and produces
        // possible next match cursors into `next_match_cursors`. After some post-processing,
        // the contents of `next_match_cursors` replenish `cur_match_cursors` and we start
        // over again.
        self.cur_match_cursors.clear();
        self.cur_match_cursors.push(MatchCursor { idx: 0, matches: self.empty_matches.clone() });

        loop {
            self.next_match_cursors.clear();
            self.black_box_match_cursors.clear();

            // Process `cur_matcher_positions` until either we have finished the input or we need to get some
            // parsing from the black-box parser done.
            let res =
                self.match_token(matcher, &parser.token, parser.approx_token_stream_pos(), track);

            // if `match_token` return Some we return that as the result. The Some might contain
            // a successful result or an error.
            if let Some(res) = res {
                return res;
            }

            // `match_token` handled all of `cur_match_cursors`, so it's empty.
            assert!(self.cur_match_cursors.is_empty());

            // `parse_tt_error` returned None, which means it couldn't conclusively produce
            // a success or an error.
            // Error messages here could be improved with links to original rules.
            match (self.next_match_cursors.len(), self.black_box_match_cursors.len()) {
                (0, 0) => {
                    // There are no possible next positions AND we aren't waiting for the black-box
                    // parser. This means this arm can't match either because no more matcher locs
                    // are left or because the next token did not match. E.g. if the arm is
                    // (a b) => {} and the invocation is m!(a b c) then c token is not expected.
                    // Another example is if arm is (a b d) and the invocation is m!(a b c) then
                    // again the c token is not expected.
                    debug!("parse_tt: error: no rules expected this token in macro call");
                    return RetryNextArmBecauseArmMatchFailed(T::build_failure(
                        parser.token.clone(),
                        parser.approx_token_stream_pos(),
                        "no rules expected this token in macro call",
                    ));
                }

                (_, 0) => {
                    // There are non-zero next_match_cursors. Dump them into the cur_match_cursors
                    // and match the next token
                    debug!("parse_tt: some next_mps but no bb_mps. Continuing next loop iteration");
                    self.cur_match_cursors.append(&mut self.next_match_cursors);
                    // Bump because the previous token matched successfully and
                    // we need to match the next token
                    parser.to_mut().bump();
                }

                (0, 1) => {
                    // There are zero next_match_cursors but exactly one black box cursor.
                    // This means we need to call the black-box parser to get some non-terminal.
                    let mut match_cursor = self.black_box_match_cursors.pop().unwrap();
                    let loc = &matcher[match_cursor.idx];
                    if let &MatcherLoc::MetaVarDecl {
                        span,
                        kind: Some(kind),
                        next_metavar,
                        seq_depth,
                        ..
                    } = loc
                    {
                        // We use the span of the meta-variable declaration to determine any
                        // edition-specific matching behavior for non-terminals.
                        let nt = match parser.to_mut().parse_nonterminal(kind) {
                            Err(mut err) => {
                                let guarantee = err.span_label(
                                    span,
                                    format!(
                                        "while parsing argument for this `{kind}` macro fragment"
                                    ),
                                )
                                .emit();
                                // If no non-terminal could be found it doesn't only terminate
                                // matching for this arm but for all arms!
                                return AbortBecauseErrorAlreadyReported(guarantee);
                            }
                            Ok(nt) => nt,
                        };
                        debug!("parse_tt: parse_non-terminal result: {nt:?})");
                        let m = match nt {
                            NtOrTt::Nt(nt) => MatchedNonterminal(Lrc::new(nt)),
                            NtOrTt::Tt(tt) => MatchedTokenTree(tt),
                        };
                        match_cursor.push_match(next_metavar, seq_depth, m);
                        match_cursor.idx += 1;
                    } else {
                        unreachable!()
                    }
                    debug!("parse_tt: cur_mps.push({match_cursor:?})");
                    self.cur_match_cursors.push(match_cursor);
                }

                (_, _) => {
                    // There are:
                    // * either more than one black box cursors and zero next match cursors
                    // * or     one black box cursor and more than zero next match cursors
                    // * or     more than one black box cursors and more than zero next match cursors
                    // In this case we don't know which one to pick so emit an ambiguity error
                    // note that this is an abort error, which means next arm won't be retried.
                    return self.ambiguity_error(matcher, parser.token.span);
                }
            }

            // We should have some match cursors to process for the next iteration
            assert!(!self.cur_match_cursors.is_empty());
        }
    }

    fn ambiguity_error<F>(
        &self,
        matcher: &[MatcherLoc],
        token_span: rustc_span::Span,
    ) -> NamedParseResult<F> {
        debug!("inside ambiguity_error");
        let nts = self
            .black_box_match_cursors
            .iter()
            .map(|mp| match &matcher[mp.idx] {
                MatcherLoc::MetaVarDecl { bind, kind: Some(kind), .. } => {
                    format!("meta-variable ${}:{}", bind, kind)
                }
                _ => unreachable!(),
            })
            .collect::<Vec<String>>()
            .join(" or ");
        let macro_name = &self.macro_name;
        let next_mps = &self.next_match_cursors;
        debug!("ambiguity_error: nts: {nts:?} for macro: {macro_name:?}. next_mps: {next_mps:?}");

        let mps = self
            .next_match_cursors
            .iter()
            .map(|mp| match &matcher[mp.idx] {
                MatcherLoc::Sequence { idx_first_after, .. } => {
                    let m = &matcher[*idx_first_after];
                    format!("{m}")
                }
                MatcherLoc::SequenceKleeneOpAfterSep { idx_first, .. } => {
                    let m = &matcher[*idx_first];
                    format!("{m}")
                }
                MatcherLoc::SequenceKleeneOpNoSep { idx_first, op } => {
                    let m = &matcher[*idx_first];
                    format!("{m} in sequence ending with {op}")
                }
                m => {
                    format!("{m}")
                }
            })
            .collect::<Vec<String>>()
            .join(" or ");

        AbortBecauseFatalError(
            token_span,
            format!(
                "ambiguity when calling macro `{}`: {}",
                self.macro_name,
                match self.next_match_cursors.len() {
                    0 => format!("token can either match {nts}."),
                    _ => format!("token can either match {nts} or {mps}"),
                }
            ),
        )
    }

    fn ensure_unique_metavariables<I: Iterator<Item = MetaVarMatch>, F>(
        &self,
        matcher: &[MatcherLoc],
        mut res: I,
    ) -> NamedParseResult<F> {
        debug!("Inside ensure_unique_metavariables");
        // Make sure that each metavar has _exactly one_ binding. If so, insert the binding into the
        // `NamedParseResult`. Otherwise, it's an error.
        let mut ret_val = FxHashMap::default();
        for loc in matcher {
            if let &MatcherLoc::MetaVarDecl { span, bind, kind, .. } = loc {
                if kind.is_some() {
                    match ret_val.entry(MacroRulesNormalizedIdent::new(bind)) {
                        Vacant(spot) => spot.insert(res.next().unwrap()),
                        Occupied(..) => {
                            return AbortBecauseFatalError(
                                span,
                                format!("duplicated bind name: {}", bind),
                            );
                        }
                    };
                } else {
                    debug!(
                        "ensure_unique_metavariables: missing fragment specifier for bind {bind}"
                    );
                    // E.g. `$e` instead of `$e:expr`, reported as a hard error if actually used.
                    // Both this check and the one in `match_token` are necessary, surprisingly.
                    return AbortBecauseFatalError(span, "missing fragment specifier".to_string());
                }
            }
        }
        ArmMatchSucceeded(ret_val)
    }
}
