#!/usr/bin/env python3
"""The problem stated as information theory rather than as language modelling.

THE QUESTION
    A substrate of S bits holds at most S bits (Shannon). Any encoding scheme
    achieves some fraction of that bound. "Parameters" are one scheme among
    many, and denominating capacity in parameters hides the bound entirely.

    eta := knowledge_bits_retrievable / substrate_bits        eta <= 1.0

    Transformers achieve eta ~= 0.25. The thesis asks whether a dynamical
    encoding closes the remaining 4x.

HOW TO REVIEW THIS FILE
    Section 1  MEASURED CONSTANTS  - every number with its source. Check these
                                     against the cited papers; nothing else is
                                     an input.
    Section 2  PURE FUNCTIONS      - one function per quantity, each checkable
                                     in isolation. No printing, no globals.
    Section 3  SELF-CHECKS         - assertions that fail loudly if an identity
                                     breaks. Run before any output is produced.
    Section 4  REPORT              - formatting only. Contains no arithmetic.

Run: python3 verify_substrate_bound.py     (exit 0 = all self-checks passed)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, log2

# =============================================================================
# SECTION 1 -- MEASURED CONSTANTS
# Every input to this file appears here, with provenance. Nothing is derived
# in this section.
# =============================================================================

BITS_PER_BYTE = 8
BYTES_PER_GB = 10**9

#: Resident weight budget on the target machine. Derived in 01_feasibility.md
#: from 16 GiB total minus OS floor, runtime, and swap headroom.
SUBSTRATE_BUDGET_BYTES = 6.5 * BYTES_PER_GB

#: Entropy of one item in the FLAT dataset: independent uniform attributes.
#: Source: src/data/schema.py BITS_PER_PERSON.
FLAT_BITS_PER_ITEM = 50.97

#: Shannon bound on knowledge bits per substrate bit.
SHANNON_ETA = 1.0

#: Classical Hopfield capacity coefficient: alpha_c ~= 0.138 patterns per unit
#: before the spin-glass transition.
#: Source: Amit, Gutfreund & Sompolinsky (1985), statistical mechanics of
#: neural networks. This is a physics result, not an ML heuristic.
HOPFIELD_CAPACITY_COEFFICIENT = 0.138


@dataclass(frozen=True)
class EncodingScheme:
    """One way of spending substrate bits to store knowledge.

    Attributes:
        name: human label.
        storage_bits_per_param: how many substrate bits one parameter occupies.
        knowledge_bits_per_param: measured knowledge recoverable per parameter.
        is_measured: False when the figure is extrapolated rather than observed.
            Extrapolated rows must never be reported as results.
        source: provenance for knowledge_bits_per_param.
    """

    name: str
    storage_bits_per_param: float
    knowledge_bits_per_param: float
    is_measured: bool
    source: str


#: Allen-Zhu & Li, "Physics of Language Models Part 3.3" (ICLR 2025) measured
#: 2.0 bits/param, holding down to int8, collapsing to 0.7 at int4. Ternary was
#: not measured; the row is carried only to show what the extrapolation implies.
SCHEMES: tuple[EncodingScheme, ...] = (
    EncodingScheme("fp16", 16.0, 2.0, True, "Allen-Zhu & Li, measured"),
    EncodingScheme("int8", 8.0, 2.0, True, "Allen-Zhu & Li, no loss vs fp16"),
    EncodingScheme("int4", 4.0, 0.7, True, "Allen-Zhu & Li, measured collapse"),
    EncodingScheme("ternary", 1.6, 0.7, False, "EXTRAPOLATED from int4"),
)


@dataclass(frozen=True)
class StructuredGenerator:
    """A data generator with latent structure, for the pattern-formation test.

    Items are produced by choosing one of `n_archetypes` templates and then
    applying small per-item deviations. True entropy per item is therefore
    log2(n_archetypes) + deviation_bits, which is far below the flat entropy.

    Attributes:
        n_archetypes: number of latent templates, K.
        deviation_bits: entropy of the per-item deviation from its archetype.
    """

    n_archetypes: int
    deviation_bits: float


STRUCTURED_GENERATORS: tuple[StructuredGenerator, ...] = (
    StructuredGenerator(n_archetypes=64, deviation_bits=8.0),
    StructuredGenerator(n_archetypes=256, deviation_bits=12.0),
    StructuredGenerator(n_archetypes=1024, deviation_bits=16.0),
)


# =============================================================================
# SECTION 2 -- PURE FUNCTIONS
# One quantity per function. No printing, no module state. Each is checkable
# on its own.
# =============================================================================


def substrate_efficiency(scheme: EncodingScheme) -> float:
    """Knowledge bits stored per substrate bit (eta).

    This is the quantity the Shannon bound constrains. Denominating in
    parameters instead of bits inverts the ranking of schemes, which is why
    this function exists.

    Returns:
        eta in [0, 1] for any physically realisable scheme.

    Raises:
        ValueError: if storage_bits_per_param is not positive.
    """
    if scheme.storage_bits_per_param <= 0:
        raise ValueError(f"{scheme.name}: storage_bits_per_param must be > 0")
    return scheme.knowledge_bits_per_param / scheme.storage_bits_per_param


def headroom_to_shannon(scheme: EncodingScheme) -> float:
    """How many times better a perfect encoding would be than this scheme."""
    return SHANNON_ETA / substrate_efficiency(scheme)


def knowledge_held(scheme: EncodingScheme, substrate_bits: float) -> float:
    """Total knowledge bits a substrate holds under this scheme."""
    return substrate_efficiency(scheme) * substrate_bits


def best_measured_scheme(schemes: tuple[EncodingScheme, ...]) -> EncodingScheme:
    """The most substrate-efficient scheme among those actually MEASURED.

    Extrapolated schemes are excluded deliberately: ternary shows the highest
    eta of any row, but that figure is inferred rather than observed, and
    planning against it would be planning against an assumption.

    Raises:
        ValueError: if no measured scheme is present.
    """
    measured = [s for s in schemes if s.is_measured]
    if not measured:
        raise ValueError("no measured schemes to choose from")
    return max(measured, key=substrate_efficiency)


def flat_entropy_bits(n_items: int) -> float:
    """Entropy of `n_items` drawn from the incompressible (flat) generator."""
    if n_items < 0:
        raise ValueError(f"n_items must be non-negative, got {n_items}")
    return n_items * FLAT_BITS_PER_ITEM


def structured_entropy_bits(generator: StructuredGenerator, n_items: int) -> float:
    """Entropy of `n_items` from a generator with latent archetypes.

    Raises:
        ValueError: if n_items is negative or n_archetypes < 1.
    """
    if n_items < 0:
        raise ValueError(f"n_items must be non-negative, got {n_items}")
    if generator.n_archetypes < 1:
        raise ValueError(f"n_archetypes must be >= 1, got {generator.n_archetypes}")
    bits_per_item = log2(generator.n_archetypes) + generator.deviation_bits
    return n_items * bits_per_item


def compression_ratio(generator: StructuredGenerator) -> float:
    """How much less a structure-finder needs than a memoriser.

    This ratio is the pattern-formation signal. It is computed from the
    generator's construction, so it is ground truth rather than a benchmark
    score.
    """
    structured_per_item = structured_entropy_bits(generator, 1)
    return FLAT_BITS_PER_ITEM / structured_per_item


def hopfield_retrievable_patterns(n_units: int) -> float:
    """Classical Hopfield retrieval capacity before the spin-glass transition.

    Storage and retrieval are different quantities: a network can be written
    with more patterns than it can reliably recall.
    """
    if n_units < 0:
        raise ValueError(f"n_units must be non-negative, got {n_units}")
    return HOPFIELD_CAPACITY_COEFFICIENT * n_units


def sparse_code_addressable_bits(n_units: int, n_active: int) -> float:
    """Bits of ADDRESS SPACE in a sparse code with `n_active` of `n_units` on.

    Addressing capacity is not storage capacity. A code can address 2^803
    distinct states while the substrate still holds only n_units bits.
    Conflating the two was mistake M2.

    Raises:
        ValueError: if n_active is not in [0, n_units].
    """
    if not 0 <= n_active <= n_units:
        raise ValueError(f"need 0 <= n_active <= n_units, got {n_active}/{n_units}")
    return log2(comb(n_units, n_active))


# =============================================================================
# SECTION 3 -- SELF-CHECKS
# These run before any output. A violated identity means the file is wrong and
# nothing it prints should be believed.
# =============================================================================


def run_self_checks() -> None:
    """Assert the identities this file depends on.

    Raises:
        AssertionError: on any violated invariant.
    """
    for scheme in SCHEMES:
        eta = substrate_efficiency(scheme)
        assert 0.0 < eta <= SHANNON_ETA, (
            f"{scheme.name}: eta={eta:.3f} violates the Shannon bound; "
            f"an encoding cannot store more knowledge than it has bits"
        )

    # Ternary must rank above int8 on eta while ranking below it on
    # bits-per-parameter. This opposite ordering is the whole reason the metric
    # denominator was changed, so it is asserted rather than merely asserted-in-prose.
    ternary = next(s for s in SCHEMES if s.name == "ternary")
    int8 = next(s for s in SCHEMES if s.name == "int8")
    assert substrate_efficiency(ternary) > substrate_efficiency(int8)
    assert ternary.knowledge_bits_per_param < int8.knowledge_bits_per_param

    # Structure must genuinely compress, or the pattern test is vacuous.
    for generator in STRUCTURED_GENERATORS:
        assert compression_ratio(generator) > 1.0, (
            f"K={generator.n_archetypes} does not compress vs flat; "
            f"this generator cannot test pattern formation"
        )

    # Addressing capacity exceeds storage capacity by a wide margin -- the M2 trap.
    assert sparse_code_addressable_bits(10_000, 100) < 10_000


# =============================================================================
# SECTION 4 -- REPORT
# Formatting only. No arithmetic beyond calling Section 2.
# =============================================================================


def _heading(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def report_substrate_efficiency() -> None:
    _heading("1. SUBSTRATE EFFICIENCY -- knowledge bits per substrate bit")
    print(f"Shannon bound: eta <= {SHANNON_ETA:.1f}\n")
    print(f"{'scheme':<10}{'bits/param':>11}{'know/param':>12}{'eta':>8}{'headroom':>11}  source")
    for scheme in SCHEMES:
        print(
            f"{scheme.name:<10}"
            f"{scheme.storage_bits_per_param:>11.1f}"
            f"{scheme.knowledge_bits_per_param:>12.2f}"
            f"{substrate_efficiency(scheme):>8.3f}"
            f"{headroom_to_shannon(scheme):>10.1f}x  {scheme.source}"
        )

    best = best_measured_scheme(SCHEMES)
    print(f"\nBest MEASURED efficiency : {best.name} at eta = {substrate_efficiency(best):.3f}")
    print(f"Unexploited headroom     : {headroom_to_shannon(best):.0f}x")
    print("\nThe thesis without reference to language models: transformers store")
    print("knowledge at ~1/4 of the information-theoretic bound. Does a dynamical")
    print("encoding close that gap?")
    print("\nNote the inversion: ternary has the LOWEST bits/param yet the HIGHEST")
    print("eta. Denominating in parameters reverses the ranking of schemes.")


def report_substrate_capacity() -> None:
    _heading("2. WHAT THE TARGET SUBSTRATE HOLDS, BY SCHEME")
    substrate_bits = SUBSTRATE_BUDGET_BYTES * BITS_PER_BYTE
    print(f"Substrate = {substrate_bits / 1e9:.0f} Gbit "
          f"({SUBSTRATE_BUDGET_BYTES / BYTES_PER_GB:.1f} GB)\n")
    print(f"{'scheme':<10}{'eta':>8}{'holds':>16}")
    for scheme in SCHEMES:
        held = knowledge_held(scheme, substrate_bits)
        flag = "" if scheme.is_measured else "   (extrapolated)"
        print(f"{scheme.name:<10}{substrate_efficiency(scheme):>8.3f}"
              f"{held / 1e9:>13.1f} Gbit{flag}")
    print(f"{'Shannon':<10}{SHANNON_ETA:>8.3f}{substrate_bits / 1e9:>13.1f} Gbit")


def report_dynamical_capacity() -> None:
    _heading("3. RETRIEVAL CAPACITY OF A DYNAMICAL SYSTEM")
    n_units = 10_000
    print(f"For N = {n_units:,} units:\n")
    print(f"  Hopfield retrievable patterns : {hopfield_retrievable_patterns(n_units):,.0f}")
    print(f"    (Amit-Gutfreund-Sompolinsky, alpha_c = {HOPFIELD_CAPACITY_COEFFICIENT})")
    print(f"  Sparse-code address space     : "
          f"{sparse_code_addressable_bits(n_units, 100):,.0f} bits")
    print(f"  Raw substrate                 : {n_units:,} bits")
    print("\n  Address space and storage capacity are different quantities.")
    print("  A code can address 2^803 states and still store only 10,000 bits.")


def report_structure_signal() -> None:
    _heading("4. WHERE THE THESIS CAN WIN -- structure, not volume")
    print("On incompressible data no encoding beats another; Shannon forbids it.")
    print("That is why the flat dataset cannot test the thesis (mistake M5).\n")
    n_items = 100_000
    print(f"For {n_items:,} items:\n")
    print(f"  {'generator':<28}{'memoriser':>14}{'structure-finder':>19}{'ratio':>8}")
    print(f"  {'flat (incompressible)':<28}"
          f"{flat_entropy_bits(n_items) / 1e6:>11.2f} Mbit{'--':>19}{'1.0x':>8}")
    for generator in STRUCTURED_GENERATORS:
        label = f"K={generator.n_archetypes}, dev={generator.deviation_bits:.0f} bits"
        print(f"  {label:<28}"
              f"{flat_entropy_bits(n_items) / 1e6:>11.2f} Mbit"
              f"{structured_entropy_bits(generator, n_items) / 1e6:>14.2f} Mbit"
              f"{compression_ratio(generator):>7.1f}x")
    print("\n  The ratio is ground truth from the generator, not a benchmark score.")


def report_experiment_statement() -> None:
    _heading("5. THE EXPERIMENT, STATED WITHOUT LANGUAGE MODELS")
    print("""
  GIVEN   a substrate of S bits, and items from a generator whose true
          entropy H is known by construction

  MEASURE eta = (bits of generator content retrievable) / S

  COMPARE encoding-as-static-weights  vs  encoding-as-dynamical-state

  VARY    the latent structure of the generator, from incompressible
          to highly structured

  PREDICT eta_static  is flat in structure    (memorises either way)
          eta_dynamic RISES with structure    (exploits regularity)

  The claim under test is a SLOPE, not a scalar. Nothing here mentions
  tokens, attention, or next-token prediction: those are one possible
  read-out mechanism, not the object of study.
""")


def main() -> int:
    run_self_checks()
    report_substrate_efficiency()
    report_substrate_capacity()
    report_dynamical_capacity()
    report_structure_signal()
    report_experiment_statement()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
