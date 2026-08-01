# Contributing to diffpes

Thank you for your interest in contributing to diffpes! This guide defines the
standards for type hints, documentation, validation, testing, and tools.

## Core Principle: Invertible Modularity

diffpes uses one differentiable pipeline in two directions. The forward
direction converts a band structure into an ARPES spectrum. The inverse
direction recovers physical parameters from measured data.

Every module is a differentiable operator. A module boundary also defines a
boundary for the inverse problem. Attach a loss at any boundary. Then solve for
the source parameters while other parameters stay fixed. This invertibility is
the primary asset of the codebase.

It rests on one invariant:

> **Reductions stay explicit, late, and differentiable. No module collapses
> information it is not forced to.**

Concretely:

- Keep matrix elements complex. Apply `|·|²` as late as possible. **Never apply
  it before a coherent sum.** Dipole channels, orbital contributions, and spin
  components interfere. An early modulus square removes circular dichroism and
  spin-ARPES observables.
- Express experimental averaging (energy/angle resolution, kz broadening,
  temperature) as explicit differentiable operations over distributions. Do
  not use hidden quadratures or fixed convolutions.
- Use `jnp.where` / `lax.cond` and continuous fields rather than discrete swaps
  or data-dependent Python control flow. This structure gives each parameter a
  derivative.
- Treat each gradient as part of the physics. Gradients form every row of the
  Fisher information matrix under the identifiability thesis. A zero, NaN, or
  conjugation error is a *physics* bug. Correct forward values do not excuse an
  incorrect gradient.

This failure mode is silent. A hard, non-differentiable, or early reduction can
leave the forward model correct. However, the reduction breaks invertibility at
one boundary. Require an explicit review justification for each such reduction.
The JAX-First rules below implement this principle.

## Development Setup

### Prerequisites

- Python 3.12–3.14 (`requires-python = ">=3.12,<3.15"`)
- [uv](https://docs.astral.sh/uv/) (package and environment manager)
- Git
- CUDA-compatible GPU (optional, for acceleration)

### Installation for Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/debangshu-mukherjee/diffpes.git
   cd diffpes
   ```

2. **Install in development mode:**
   ```bash
   # Everything (docs, tests, notebooks, dev tooling)
   uv sync --extra dev

   # With CUDA support as well
   uv sync --extra dev_cuda
   ```

   The `pyproject.toml` file defines these groups: `docs`, `test`, `notebooks`,
   `cuda`, `dev`, `dev_cuda`, and `all`. The `dev` group includes documentation,
   tests, notebooks, and tools. The `dev_cuda` group adds CUDA support.

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Project Structure

```
diffpes/
├── src/diffpes/           # Main source code
│   ├── certify/           # JAX-native forward certification, evidence,
│   │                      #   provenance, policies, and information flow
│   ├── inout/             # Data I/O (POSCAR, EIGENVAL, KPOINTS, DOSCAR,
│   │                      #   PROCAR, CHGCAR, HDF5, certificates) and plotting
│   ├── maths/             # Dipole selection rules, spherical harmonics, Gaunt
│   ├── radial/            # Radial primitives (Bessel, wavefunctions, integrals)
│   ├── simul/             # ARPES forward model (matrix elements, polarization,
│   │                      #   self-energy, broadening, resolution, spectra)
│   ├── tightb/            # Tight-binding models (Hamiltonians, diagonalization,
│   │                      #   projections)
│   ├── types/             # Equinox PyTree types, factories, and aliases
│   └── utils/             # Mathematical utilities
├── tests/                 # Test suite (mirrors src layout, see below)
└── docs/                  # Sphinx documentation
```

Each subpackage exposes its public API through `__init__.py` with an explicit
`__all__`. The top-level `src/diffpes/__init__.py` enables 64-bit precision
(`jax.config.update("jax_enable_x64", True)`) and sets CPU threading XLA flags
**before** any module imports JAX. Keep import-time side effects confined to that
module.

The repository contains the changelog, documentation, pre-commit configuration,
and CI workflows.

## Coding Standards

### JAX-First Development

diffpes uses JAX for differentiable, high-performance computation. All
new code must follow JAX best practices:

**Required JAX Patterns:**
- Use `jax.lax.scan` instead of Python `for` loops over array data
- Use `jax.lax.cond` / `jnp.where` instead of data-dependent `if`/`else`
- Use `.at[].set()` for array updates instead of in-place modification
- Keep functions purely functional — no side effects, no global mutable state
- Code must remain traceable for `jit`, `grad`, `vmap`, and sharding

**Solver stack:** Use [Optimistix](https://docs.kidger.site/optimistix/) for
optimization and nonlinear solves. Its methods include `least_squares`,
`root_find`, `fixed_point`, and `minimise`. Use
`optimistix.OptaxMinimiser` for optax optimizers. Use
[Lineax](https://docs.kidger.site/lineax/) for linear solves. Use the implicit
differentiation tools from these libraries. Reserve a custom `custom_vjp` for
unsupported primitives, such as regularized `eigh` gradients.

**Differentiability rules of the house:**
- Require `jax.grad` to agree with central finite differences. A finite gradient
  is not necessarily correct. Reject a zero gradient when the physics has real
  sensitivity.
- Guard against the double-`jnp.where` NaN trap.
- Sanitize the unsafe branch input with an inner `where` when a branch can
  produce `nan` or `inf`. Do not only sanitize the output.
- Carry complex *parameters* as stacked real values. Convert them to complex
  values inside the forward model. Keep complex *state*, such as matrix
  elements and eigenvectors, complex. Apply `|·|²` late.
- `jnp.linalg.eigh` gradients blow up at degeneracies (symmetry points, Kramers
  pairs under SOC). Differentiate only gauge-invariant combinations, such as
  projectors and spectral functions. Use the degeneracy-aware tools in
  `diffpes.tightb`. Alternatively, use a Green's-function formulation. Never
  differentiate raw eigenvectors at a possible degeneracy.

**Example:**
```python
# ❌ Wrong - Python loops and conditionals over array data
def bad_function(x):
    result = []
    for i in range(len(x)):
        if x[i] > 0:
            result.append(x[i] * 2)
    return jnp.array(result)


# ✅ Correct - vectorized JAX
@jaxtyped(typechecker=beartype)
def good_function(x: Float[Array, " n"]) -> Float[Array, " n"]:
    doubled_positive: Float[Array, " n"] = jnp.where(x > 0, x * 2, x)
    return doubled_positive
```

### Type Hinting with jaxtyping and beartype

Every public function is runtime-typechecked with the
`@jaxtyped(typechecker=beartype)` decorator stack and annotated with
`jaxtyping` shape/dtype specs:

```python
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Complex, Float, jaxtyped

from diffpes.types import ArpesSpectrum, BandStructure, scalar_float


@jaxtyped(typechecker=beartype)
def simulate_spectrum(
    bands: BandStructure,
    photon_energy: scalar_float,
    temperature: Optional[scalar_float] = 15.0,
) -> ArpesSpectrum:
    """..."""
```

#### Type Hinting Rules:
- Annotate all parameters and return values.
- Use `beartype.typing.Tuple[...]` for multiple returns.
- Annotate intermediate variables inside function bodies too — e.g.
  `theta_rad: Float[Array, ""] = jnp.deg2rad(theta_deg)`.
- **Assign before returning.** Bind a function's result to a type-annotated
  variable. Return that name instead of a bare expression. This rule gives the
  result an explicit type at its definition.
- Use descriptive dimension names in shape specs:
  `Float[Array, "nkpt nband"]`, `Complex[Array, "nkpt nband natom norb"]`,
  scalars as `Float[Array, ""]`.
- **Never annotate with a bare `np.ndarray`.** That annotation states no dtype
  and no shape. Import `NDArray` from `numpy.typing`. Then use `NDArray` as the
  array type inside a jaxtyping spec: `Float[NDArray, "m n p"]`. This form gives
  a NumPy array the same dtype and shape checks as a JAX `Array`. Apply the rule
  to every position, which includes parameters, return types, intermediate
  variables, and nested types such as
  `Tuple[Float[NDArray, " n"], Float[NDArray, " n"]]`.

  Take `NDArray` from `numpy.typing` only. Do not write
  `from numpy import ndarray as NDArray`. That alias needs a `# noqa: N812`
  suppression, and the repository contains no `N812` suppression. A bare
  `NDArray` is also invalid: beartype rejects its unbound `_ScalarT` type
  variable. Always give `NDArray` a jaxtyping dtype and shape.
  Do not use the NumPy parameterization `NDArray[np.float64]` either.

  ```python
  import numpy as np
  from beartype import beartype
  from jaxtyping import Float, Int, jaxtyped
  from numpy.typing import NDArray


  # ❌ Wrong - no dtype and no shape
  def bad(nodes: np.ndarray, counts: np.ndarray) -> np.ndarray: ...


  # ✅ Correct - dtype and shape are explicit
  @jaxtyped(typechecker=beartype)
  def good(
      nodes: Float[NDArray, " n"],
      counts: Int[NDArray, " n"],
  ) -> Float[NDArray, " n"]:
      weighted: Float[NDArray, " n"] = nodes * counts
      return weighted
  ```
- Prefer the scalar aliases from `diffpes.types` (`types/aliases.py`) for
  scalar arguments. These unions accept Python scalars and zero-dimensional JAX
  arrays.
- Import shared types from `diffpes.types`, not by re-defining them.
- **Cross-subpackage imports are public and go through the subpackage.**
  Apply two requirements to each cross-subpackage import. First, the source
  subpackage must export the name publicly. The module `Routine Listings`,
  module `__all__`, and package `__init__.py` must contain the name. Second,
  import the name from the subpackage, not from one of its files. For example,
  use `from diffpes.types import KB_EV_PER_K`. Do not import it from
  `diffpes.types.constants`. A name that another subpackage needs is public.
  Promote such a name. Use deep relative imports only within one subpackage.
- **Never rename diffpes names on import** (`import ... as`). Do not use
  `KB_EV_PER_K as _KB` or `_N_ORBITALS as _NORBS`. An alias creates a
  second name for one constant. This extra name complicates searches and
  reviews. Community-standard aliases such as `jnp`, `np`, and `plt` are
  exceptions. Import `NDArray` from `numpy.typing`. Do not rename
  `numpy.ndarray` to `NDArray` with an alias.
- Import typing constructs (`Optional`, `Union`, `Tuple`, `List`, `Dict`,
  `TypeAlias`) from `beartype.typing`, not the stdlib `typing` module.

### Custom Types and PyTrees

**All types live in `diffpes.types`. There are no exceptions.** Define every
PyTree, carrier, type alias, and `make_*` factory under `src/diffpes/types/`.
Other subpackages import these types from `diffpes.types`. They do not define
local PyTrees, containers, or factories.

This rule gives each type one import surface, one registration, and one
validation contract. It also prevents duplicate carriers. The fitting layer
compares `ArpesSpectrum` objects, so the project must define one
`ArpesSpectrum` type. A result container or parameter container is also a type.
Define it in `diffpes.types`, not beside its producer.

Use **Equinox modules** (`eqx.Module`) for structured data types. These
immutable JAX PyTrees flow through `jit`, `grad`, and `vmap`. Declare static,
non-array metadata fields with `eqx.field(static=True)`. JAX then excludes
these fields from the differentiable leaves.

```python
import equinox as eqx
from jaxtyping import Array, Float


class BandStructure(eqx.Module):
    """JAX-compatible band structure with k-points and eigenvalues.

    :see: :class:`~.test_bands.TestBandStructure`
    ...
    """

    kpoints: Float[Array, "nkpt 3"]
    eigenvalues: Float[Array, "nkpt nband"]
    fermi_energy: Float[Array, ""]
```

### Validation Pattern for Factory Functions

Construct custom types through `make_*` factory functions that validate
inputs. Put these factories in `diffpes.types` **next to the type that they
build**. Never put a factory in the consuming subpackage. Use a two-tier
approach:

- Use plain Python `raise ValueError` for **static shape and structure checks**
  that JAX can resolve at trace time.
- **Data-dependent (traced) checks** use `equinox.error_if`, which raises at
  runtime without breaking `jit`.

```python
@jaxtyped(typechecker=beartype)
def make_band_structure(
    kpoints: Float[Array, "nkpt 3"],
    eigenvalues: Float[Array, "nkpt nband"],
    fermi_energy: scalar_float,
) -> BandStructure:
    """Create a BandStructure PyTree with data validation.

    :see: :class:`~.test_bands.TestBandStructure`
    ...
    """
    kpoints = jnp.asarray(kpoints)
    eigenvalues = jnp.asarray(eigenvalues)

    if eigenvalues.shape[0] != kpoints.shape[0]:   # static -> ValueError
        raise ValueError("eigenvalues and kpoints disagree on nkpt")

    checked_eigenvalues = eqx.error_if(            # traced -> eqx.error_if
        eigenvalues,
        jnp.any(jnp.isnan(eigenvalues)),
        "eigenvalues must be finite",
    )
    band_structure: BandStructure = BandStructure(
        kpoints=kpoints,
        eigenvalues=checked_eigenvalues,
        fermi_energy=jnp.asarray(fermi_energy),
    )
    return band_structure
```

### Units, Conventions, and Indexing

- Use **eV** for energies, **Angstrom** for lengths, and **1/Angstrom** for
  k-vectors. Use degrees for angles at public API boundaries. Convert each
  angle once to an annotated radian value inside the boundary.
- Use standard Python/NumPy indexing everywhere (zero-based, end-exclusive):
  non-s orbitals are `slice(1, 9)`, p orbitals `slice(1, 4)`, d orbitals
  `slice(4, 9)`. Do not use MATLAB-style indexing notation, even in comments.
- Use the sign and phase conventions from the physics canon. These conventions
  include spherical-harmonic phases, Gaunt coefficients, polarization vectors,
  and rotation frames. Treat a difference between code and canon as a bug. Do
  not define a different local convention.

### Naming: Domain Terms Only

Name every object for the concept that it represents. Use domain terms from
physics, mathematics, or software structure.

**Never put project-management vocabulary in a name.** This rule excludes plan
numbers, work-package numbers, gate identifiers, phase labels, sprint labels,
and milestone labels. Development tracking stays in the separate planning
repository. It is immaterial to a user of this library.

The rule applies to all of these:

- file and directory names;
- module, class, function, and variable names;
- test module, test class, and test function names;
- reference-data artifacts and their manifest schema strings;
- docstrings, inline comments, and Markdown prose;
- certification owner and transformation identities.

```python
# ❌ Wrong - tracking vocabulary leaks into the repository
# file: tests/test_diffpes/test_simul/test_plan06_g15.py
def test_plan07_wp3_carrier() -> None:
    """Validate the Plan-06 gate 06.G15 covariance requirement."""


# ✅ Correct - the name states the content
# file: tests/test_diffpes/test_simul/test_complete_shell_covariance.py
def test_complete_shell_covariance() -> None:
    """Validate rotational covariance across a complete p or d shell."""
```

Name a verification requirement for the property that it checks. Write
`chinook-tightbinding-parity`. Do not write `04.G6`. A reader must understand
the requirement without access to a tracking document.

Name a certification owner for its scientific domain. Write
`org.diffpes.matrixel`. Do not write `org.diffpes.plan.06`.

### Documentation Standards

Docstrings follow the **NumPy / numpydoc convention**. Ruff and `pydoclint`
enforce this convention through `pyproject.toml`. The `interrogate` tool checks
coverage with `fail-under = 90`. Do **not** invent section headers. Use the
numpydoc sections and three project extensions. Modules use `Extended Summary`
and `Routine Listings`. Functions use `Implementation Logic`.

The project sets the pydoclint option `check-return-types = false`. Jaxtyping
shape strings
(e.g. `Float[Array, "nkpt nband"]`) are core signature syntax that pydoclint
cannot reliably parse for return-type comparison. Do not degrade a correct
jaxtyping annotation to satisfy that comparison. Argument types and order,
plus required `Returns` and `Yields` sections, remain mandatory.

#### Prose Style: Simplified Technical English (ASD-STE100)

All repository prose conforms to
[ASD-STE100 Simplified Technical English](https://www.asd-ste100.org/).
This scope includes every docstring, Markdown file, and tutorial Markdown cell.
STE reduces ambiguity in technical text. Apply these primary rules:

- **Keep sentences short.** Maximum 20 words for an instruction, 25 for a
  description. One topic per sentence. One instruction per sentence.
- **Use the active voice and name the agent.** Write "The function computes
  the spectrum." Do not use a passive form.
- **Use the present tense for descriptions** and the imperative for
  instructions ("Compute the weights", "Do not use the recursion").
- **One term, one meaning.** Use the same word for one concept everywhere.
  Do not alternate between "compute", "calculate", and "evaluate" for one
  operation. The verbatim summary rule enforces this practice for API
  descriptions.
- **Keep the articles.** Write "the eigenvalues of the Hamiltonian", not
  telegraph-style "eigenvalues of Hamiltonian".
- **Do not use noun clusters longer than three nouns.** Add prepositions to
  separate the nouns. Write "the width of the Voigt profile."
- **Do not use idioms or figures of speech.** These forms can cause ambiguous
  translations and instructions.
- **Technical names are exempt.** Domain terms (ARPES, PyTree, Kramers
  doublet, Gaunt coefficient, `jnp.where`) are STE technical names. Use one
  spelling consistently for each technical name.

Reviewers check full STE dictionary compliance because tools cannot check it.
ASD provides the specification after free registration at asd-ste100.org.
Simplify each sentence that a reviewer identifies as non-STE.

#### Module Docstrings

Start each module with a one-line summary and an `Extended Summary`. Add a
`Routine Listings` section that references every public object. Add a `Notes`
section when it is relevant. In each package `__init__.py`, list every
submodule in the `Extended Summary`. Use a `- :mod:`name`` entry and one
description for each submodule. Update this list when you add a submodule.

```python
# src/diffpes/radial/__init__.py
"""Differentiable radial primitives for photoemission matrix elements.

Extended Summary
----------------
This subpackage provides the radial building blocks of the matrix-element
engine: spherical Bessel functions, bound and continuum radial
wavefunctions, and the quadrature that contracts them into radial
integrals.

The submodules are organized as follows:

- :mod:`bessel`
    Spherical Bessel functions in JAX.
- :mod:`integrate`
    Radial quadrature for matrix-element integrals.
- :mod:`wavefunctions`
    Bound and continuum radial wavefunctions.

Routine Listings
----------------
:func:`radial_integral`
    Contract radial wavefunctions against the final state.
:func:`spherical_bessel_jl`
    Evaluate spherical Bessel function j_l(x).
"""
```

Copy each symbol summary verbatim into its `Routine Listings` entry. Copy each
submodule summary verbatim into its `- :mod:` entry.

Use the correct Sphinx role in `Routine Listings`. Use `:func:` for functions
and `:class:` for classes or PyTrees. Use `:obj:` for aliases or constants.
Use `:mod:` for submodules.

**List every public object in three places, and keep all three synchronized:**

1. List the object in its module-level `Routine Listings` section.
2. List the object in that module's `__all__` value.
3. Repeat the object in the subpackage `Routine Listings` and `__all__`.

A symbol that is absent from one location is a defect. Update all three
locations when you add, rename, or remove a public function. Keep the summary
identical in the function docstring and both `Routine Listings` sections.

**Export each symbol once from its owning module.** Expose that module through
its subpackage `__init__.py`. Never add a second export for convenience or
compatibility. When a symbol moves, update every import and delete the old path
in one change. Do not add a shim, alias, or `DeprecationWarning`. Record the
migration only in `CHANGELOG.md`. This rule is the zero-legacy policy.

#### Function and Class Docstrings

Every function docstring answers three questions in Simplified Technical
English. The `Parameters` section states what the function accepts. The
`Returns` section states what the function produces. The `Implementation
Logic` section states how the function works. A one-formula function can use
`Notes` for this information. A docstring is incomplete if it omits one answer.

```python
@jaxtyped(typechecker=beartype)
def free_electron_kz(
    kinetic_energy: scalar_float,
    kpar: Float[Array, " n"],
    inner_potential: scalar_float,
) -> Float[Array, " n"]:
    r"""
    Calculate out-of-plane momentum in the free-electron final state.

    Computes :math:`k_z` from the photoelectron kinetic energy and
    in-plane momentum under the free-electron final-state approximation
    with inner potential :math:`V_0`.

    :see: :class:`~.test_kinematics.TestFreeElectronKz`

    Parameters
    ----------
    kinetic_energy : scalar_float
        Photoelectron kinetic energy in eV.
    kpar : Float[Array, " n"]
        In-plane momentum magnitudes in 1/Angstrom.
    inner_potential : scalar_float
        Inner potential :math:`V_0` in eV.

    Returns
    -------
    kz_values : Float[Array, " n"]
        Out-of-plane momenta in 1/Angstrom.

    Notes
    -----
    1. Form the free-electron dispersion prefactor :math:`2m/\hbar^2`.
    2. Evaluate :math:`k_z = \sqrt{(2m/\hbar^2)(E_k + V_0) - k_\parallel^2}`.
    3. Bind the result to ``kz_values`` and return it.

    See Also
    --------
    kinetic_energy_from_photon : Kinetic energy from photon energy and
        work function.
    """
```

The `Returns` entry uses the name `kz_values`. The function body returns this
type-annotated variable. Thus, the docstring, body, and signature agree.

##### Section order

A function docstring uses these sections, in this order, omitting any that
do not apply:

1. Summary line
2. Extended summary (untitled prose, directly after the summary)
3. `:see:` test cross-reference
4. `Implementation Logic`
5. `Parameters`
6. `Returns` (or `Yields` for generators)
7. `Raises`
8. `Notes`
9. `References`
10. `See Also`
11. `Examples`

##### Summary line

- Write one imperative sentence that ends in a period and fits on one line.
  Write "Compute normalized Gaussian broadening profile." Do not write "This
  function computes." Do not repeat the parameter list.
- Copy this exact sentence **verbatim** into both `Routine Listings` sections.
  One section belongs to the module. The other belongs to the subpackage
  `__init__.py`. This requirement is the three-places rule. Update all three
  locations when the sentence changes.

##### Extended summary

- Use one or two short paragraphs to describe the quantity and its regime.
  State the approximation and its domain of validity. For example, state the
  accuracy of the pseudo-Voigt method. Use `:math:` for inline equations.
  Put process information in `Implementation Logic`, not in this summary.

##### `:see:` cross-reference

- Give every public object a `:see:` link to its test class. Give that test
  class a link back to the object. Update both links when either name changes.

##### `Implementation Logic` (house section)

- Add this section when a function does more than transcribe one formula.
  A short function can put its process in `Notes`.
- Use numbered bold steps. Start each step with a `::` literal block that
  quotes the actual expressions. After the block, explain the reason for the
  step with indented prose.

  ```
  1. **Compute normalization factor**::

         norm_factor = sqrt(2 * pi) * sigma

     This prefactor ensures the profile integrates to unity.
  ```
- Keep the steps synchronized with the function body. Reviewers compare the
  steps with the code. Stale `Implementation Logic` is a defect.

##### `Parameters`

- Add one entry for each signature parameter. Keep signature order. Use the
  numpydoc `name : type` form. Copy the annotated type exactly.
- **Give units for every physical quantity.** State eV for photon energy and
  1/Angstrom for momentum. State degrees or radians for every angle.
- State defaults in prose: "Default 15.0."
- **Mark static, non-traced parameters.** This group includes
  `static_argnames` and Python values that control shapes or flow. It also
  includes values in `eqx.field(static=True)`. State that changing the value
  causes retracing.
- Name each PyTree type with a `:class:` reference. Do **not** repeat its field
  documentation. Document the fields only on the type.

##### `Returns` / `Yields`

- Name each return value after the **type-annotated variable that the body
  returns**. Use `name : type` and state the units. Give each tuple element a
  named entry in order.

##### `Raises`

- Document **every explicit raise**. Use `ValueError` for static validation and
  state the failed condition. Use `EquinoxRuntimeError` for traced
  `eqx.error_if` checks and state the runtime condition. Do not document
  beartype or jaxtyping rejections.

##### `Notes`

- State the physics limitations and approximation limits that a user needs.
  Reference the physics canon for conventions. Do not define a local
  convention.
- **Add differentiability notes when they are relevant.** Identify parameters
  that carry gradients. Describe `safe_*` guards at boundary rays. State each
  known zero-gradient plateau. A documented zero gradient is a limitation. An
  undocumented zero gradient is a bug.

##### `References`

- Numpydoc footnotes (`.. [1] Author, "Title", Journal Vol, pages
  (year).`), cited in the text as `[1]_`.
- **Use unique footnote labels across the module.** The `automodule` directive
  renders all module docstrings on one page. Duplicate `.. [1]` labels collide.
  Continue the numbering across functions.

##### `See Also`

- List related public functions as `name : one-line description`. Use the
  target's summary verbatim when it fits.

##### `Examples`

- Doctest format, deterministic, cheap (CPU, small arrays). Ruff formats
  doctest code (`docstring-code-format`), and the rendered docs display it.

##### Class docstrings (`eqx.Module` PyTrees)

```python
class LifetimeModel(eqx.Module):
    """Configure energy-dependent lifetime broadening.

    Carries the parameters of the imaginary self-energy
    :math:`\\Gamma(E)` used to build Lorentzian linewidths.

    :see: :class:`~.test_lifetime.TestLifetimeModel`

    Attributes
    ----------
    gamma_0 : Float[Array, ""]
        Constant offset of :math:`\\Gamma(E)` in eV.
    mode : str
        Evaluation mode selector (**static** — stored via
        ``eqx.field(static=True)``; changing it triggers retracing).

    See Also
    --------
    make_lifetime_model : Validated factory for this type.
    """
```

- Summary line and extended summary follow the same rules as functions;
  `:see:` points at the type's test class.
- Document every field in `Attributes` and keep declaration order. Use
  `name : type` and state units. Mark each `eqx.field(static=True)` field as
  **static**.
- Do not add an `__init__` docstring. Equinox generates the constructor.
  Document the construction contract on the `make_*` factory. Name that
  factory in `See Also`.
- `Methods` section only if the class exposes public methods.

##### Factory (`make_*`) docstrings

- Use the factory docstring as the validation contract. Identify static
  `ValueError` checks and traced `eqx.error_if` checks. Repeat both categories
  in `Raises`. Name the constructed variable in `Returns`.

##### Private objects and raw strings

- Give each `_`-prefixed helper at least a summary. Give numerical helpers full
  `Parameters`, `Returns`, and `Notes` sections. Private code is exempt only
  from the three-places rule.
- Use a raw string (`r"""`) when a docstring contains a backslash.

### Code Style

Ruff enforces the style (`line-length = 79`, `target-version = "py312"`,
double quotes). The active lint rule set includes `D, E, F, B, I, N, UP, ANN,
S, A, C4, PIE, PT, RET, SIM, ARG, ERA, PL`. Key conventions:

- **Variable Names**: descriptive `snake_case`; long names over abbreviations
  (`photoemission_intensity`, not `pi`). Scientific single-letter symbols
  (`G`, `L`, `S`) can mirror the physics.
- **Do not use inline comments in `src/` unless they are necessary.** Put the
  explanation in the docstring. Use `Parameters`, `Returns`, and
  `Implementation Logic` to explain the function. Tool directives are valid
  comments. A one-line reason is also valid when a docstring cannot contain
  it. Delete a comment that only describes the next line.
- **Pure functions**: no side effects; return new data.
- **Imports**: sorted by isort (`I`); imports inside functions only to guard
  optional dependencies or platform branches.

## Testing

The test suite uses `pytest` with `chex`, `pytest-cov`, and `pytest-xdist`.
Property-based tests use `hypothesis`. Tests follow the same type and docstring
rules as `src/`. Every test method returns `None` and uses annotated
intermediates. Its docstring states what the test verifies and how it verifies
that property.

### Test Layout

Tests mirror the source layout under `tests/test_diffpes/`:

```
tests/
└── test_diffpes/
    ├── test_inout/test_chgcar.py
    ├── test_inout/test_doscar.py
    ├── test_inout/test_eigenval.py
    ├── test_inout/test_kpoints.py
    ├── test_inout/test_poscar.py
    ├── test_inout/test_procar.py
    ├── test_maths/test_gaunt.py
    ├── test_radial/test_bessel.py
    ├── test_simul/...
    ├── test_tightb/test_hamiltonian.py
    └── test_types/...                 # one test_<module>.py per source module
```

- Name test files `test_<module>.py`.
- Name test classes `Test*` and usually inherit from `chex.TestCase`.
- Name test functions `test_*`.
- One `Test<Symbol>` class per public symbol, carrying the `:see:`
  back-reference to the symbol under test.

### What a Test Must Validate Against

**Use external truths, never diffpes outputs.** Compare verification results
against a closed-form result, a `scipy` or `sympy` reference value, or a
published number. Closed-form examples include hydrogenic radial integrals,
Rashba spinors, and free-electron kinematics. Do not use a stored diffpes
output or an unverified magic number.

**Physics is the oracle. chinook is not.** A pinned chinook artifact is a
cross-check against an independent implementation. It does not define
correctness. When chinook and the physics canon disagree, use C-type evidence
to resolve the dispute. This evidence includes analytic results, invariants,
normative formats, and independent convergence. Amend the canon if necessary.
Record each disagreement and its resolution beside the artifact.

**Never import chinook in the test suite.** Tests read pinned chinook values
from committed artifacts in `tests/data/`. Chinook-importing generators live
only outside the DiffPES source and test trees, in a separate repository, and
run manually in a separate pinned environment. Only
immutable data, hashes, and provenance cross into DiffPES. Do not add chinook
to any dependency group. Do not import or invoke it from source, tests,
conftest, fixtures, helpers, or CI. Keep generator Python outside `tests/`.
Never use a conditional skip based on Chinook availability. Tests
must read each comparison value from a committed artifact, not an inline magic
number. Repository-floor checks and the pytest import firewall enforce this
one-way boundary.

**Test gradients explicitly.** Give every differentiable primitive a central
finite-difference test with a stated tolerance. Add a zero-gradient tripwire
for every parameter that must carry sensitivity. Forward-value tests cannot
detect a corrupted Fisher row.

**Writing tests:**
- Prefer `chex` assertions over bare `assert` for arrays:
  `chex.assert_shape`, `chex.assert_trees_all_close`,
  `chex.assert_tree_all_finite`.
- Use parameterized cases for convention-sensitive code, including signs,
  phases, and orbital orderings.
- Use `hypothesis` for invariants, including unitarity, sum rules, and gauge
  invariance under random eigenvector phases.
- Test the relevant `jit`, `grad`, and `vmap` paths explicitly.

### Test Code Conventions

Tests are first-class source. Apply the `src/` style rules with the following
test-specific adaptations:

- **Type-hint test bodies and helpers exactly as in `src/`.** Every test
  method uses `def test_*(self) -> None:`. Annotate intermediate variables.
  Apply the assign-before-returning rule to every helper that returns data.
  Give shared helpers complete `jaxtyping` annotations. Apply
  `@jaxtyped(typechecker=beartype)` when arrays flow through a helper.
- **Document *what* and *how* on every test, class, and module (numpydoc).**
  Treat a test docstring as a specification. Start with an imperative summary
  line. In `Extended Summary`, state the verified property, invariant, or
  expected value. Include applicable units and tolerances. In `Notes`,
  describe the inputs, fixtures, assertion strategy, and relevant JAX
  transformations. Make each module docstring summarize the file's coverage.
  Make each **`Test<Symbol>` class** docstring name the symbol and case scope.
- **Publish test docstrings as documentation.** Sphinx renders the test suite
  as a *Testing / Validation* reference. These
  docstrings explain library guarantees and their verification methods. Write
  them as reader-facing prose. Use paired `:see:` cross-references to connect
  source and test documentation in both directions.
- **No `__all__` or `Routine Listings` in test modules.** Tests are not a
  public API, so the three-places rule does **not** apply. Give each test
  module a one-line summary and an extended summary.
- **Prefix private helpers with `_` and keep them local.** Put reused fixtures
  in the shared helper modules. Use `tests/_factories.py`,
  `tests/_assertions.py`, or `tests/_types.py`. Do not copy shared fixtures
  across files.

Example:

```python
import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

import diffpes


class TestFermiFunction(chex.TestCase):
    """Validate :func:`~diffpes.simul.fermi_function`.

    Covers the Fermi-Dirac occupation across the ARPES temperature range:
    known-value accuracy at the Fermi level, shape, and finiteness.

    :see: :func:`~diffpes.simul.fermi_function`
    """

    def test_half_occupation_at_fermi_level(self) -> None:
        """Occupation at the Fermi level equals 1/2 at any temperature.

        Confirms ``fermi_function`` reproduces the analytic Fermi-Dirac
        value :math:`f(E_F) = 1/2` independent of temperature (the *what*).

        Notes
        -----
        Evaluates ``fermi_function`` at ``E = E_F = 0`` eV for temperatures
        [10, 100, 300] K and asserts shape ``(3,)``, finiteness, and
        closeness to 0.5 at ``rtol=1e-12`` (the *how*).
        """
        temperatures: Float[Array, "3"] = jnp.array([10.0, 100.0, 300.0])
        occupations: Float[Array, "3"] = diffpes.simul.fermi_function(
            energy=jnp.zeros(3),
            fermi_energy=0.0,
            temperature=temperatures,
        )

        chex.assert_shape(occupations, (3,))
        chex.assert_tree_all_finite(occupations)
        chex.assert_trees_all_close(occupations, 0.5 * jnp.ones(3), rtol=1e-12)
```

Match each `:see:` pair. Make the source symbol point to `Test<Symbol>` and
the test class point to the symbol. Add both references in the same change.
Update both references when either target changes.

### Running Tests

```bash
# Run the whole suite
pytest

# Run a single module / class / test
pytest tests/test_diffpes/test_radial/test_bessel.py
pytest tests/test_diffpes/test_radial/test_bessel.py::TestSphericalBesselJl

# Coverage
pytest tests/ --cov=src/diffpes --cov-report=term-missing
```

## Tutorial Notebooks

Tutorials use paired Jupyter notebooks and Jupytext percent scripts in
`tutorials/`. The pair contains an `.ipynb` file and a `.py` file. This
format keeps source differences reviewable. **Put explanations in Markdown
cells, not code comments.** Put narrative, motivation, and physics in
Markdown blocks. Keep code cells free of comments. Apply these ASD-STE100
rules to all Markdown cells. After editing a pair, synchronize it and remove
outputs before committing. The pre-commit hooks perform these actions.

## Pull Request Process

### Before Submitting

```bash
# Lint and format (must match CI)
ruff check src/ tests/
ruff format src/ tests/

# Source docstring structure
pydoclint src/

# Type check
ty check

# Run all pre-commit hooks
pre-commit run --all-files

# Run the test suite
pytest
```

`ty` is the project's type checker. `pre-commit` runs ruff checks, ruff
formatting, and the other hooks. If a hook modifies files, stage the changes
and commit again.

Pre-commit hooks generate two files. **Do not edit these files manually.**
The files are `.github/badges/loc.json` and `requirements.txt`. The first file
contains badge data for the line count. The second file exports `uv.lock` for
the GitHub dependency graph. GitHub does not yet read `uv.lock` directly.
Both files regenerate locally during a commit. CI does not write them.

### PR Guidelines

1. **Branch Naming:** Use a descriptive name, such as `feature/slab-hamiltonian` or
   `fix/gaunt-phase-convention`.
2. **Commit Messages:** Write a clear summary line, then bullet points for the
   substantive changes (implementation, tests, docs).
3. **PR Description:** State the purpose, reason, test method, and breaking
   changes.

### Review Process

All PRs require:
- [ ] Passing CI tests
- [ ] Code review approval
- [ ] Documentation updates (if applicable)
- [ ] No merge conflicts

## Issue Guidelines

**Bug reports:** Include a minimal reproducible example, expected and actual
behavior, environment details, and error messages. Environment details include
the Python version, JAX version, and processor type. For a *wrong-gradient*
bug, include the finite-difference comparison. Treat a wrong gradient as a
wrong forward value.

**Feature requests:** Include the use case, proposed API, performance
considerations, and relationship to existing functionality. Omit the proposed
API when it does not apply.

## Development Guidelines

### Adding New Features

1. **Design Phase:**
   - Discuss the approach in an issue first.
   - Follow the pinned conventions and verification gates for that area.
   - Consider JAX constraints (tracing, shapes, purity, degeneracies) early.
   - Plan the type signatures, custom types, and public API.

2. **Implementation:**
   - **Put every new type, PyTree, or `make_*` factory in `diffpes.types`.**
   - Import the new object into the consuming subpackage.
   - Place other code in the appropriate subpackage.
   - Export public code through the package's `__init__.py` and `__all__`.
   - Maintain the three-places rule.
   - Decorate with `@jaxtyped(typechecker=beartype)` and annotate fully.
   - Add numpydoc docstrings with a `:see:` cross-reference to the tests.
   - Mirror the source path under `tests/test_diffpes/`.
   - Add external-truth and gradient finite-difference gates.

3. **Documentation:**
   - Update API documentation and `Routine Listings`.
   - Add a tutorial example if it introduces user-facing functionality.
   - Note behavior changes in `CHANGELOG.md`.

### API Evolution (zero-legacy)

The codebase has **no compatibility layer**. When an API changes:

- Add **no shims, aliases, re-exports, or `DeprecationWarning`s** for old
  import paths or signatures.
- Update every call site and **delete** the old path in the same change.
- Never ship two implementations or import paths together.
- The **only** migration record is a `CHANGELOG.md` note.
- Prefer a correct API over preserving an incorrect one. Pre-1.0 releases can
  contain breaking changes.

### Versioning

`[project].version` in `pyproject.toml` is the **single source of truth** for
the package version. Use CalVer, such as `2026.06.01`. PEP 440 normalizes this
example to `2026.6.1` in built artifacts.

### Building and Releasing

Use **uv for the complete packaging process**. The build backend is `uv_build`
in the `pyproject.toml` `[build-system]` table. Publish releases with
`uv publish`. Do not use `setuptools`, `build`, or `twine`.

```bash
# Build the sdist and wheel into dist/
uv build

# Sanity-check the artifacts
python -m zipfile -l dist/diffpes-*.whl

# Publish to PyPI (uses a PyPI API token)
UV_PUBLISH_TOKEN=<pypi-token> uv publish
```

Release checklist:

1. Update `[project].version` with CalVer and update `CHANGELOG.md` in the same
   commit.
2. Run `ruff check src/ tests/`, `pydoclint src/`, `ty check`, and `pytest` on
   the release commit.
3. Run `uv build` from a clean tree.
4. Verify that the wheel contains the complete `diffpes/` package.
5. Verify that its metadata contains `License-Expression: MIT`.
6. Tag the release commit with `v<version>`.
7. Run `uv publish`.

## Getting Help

- **Questions:** Open a discussion or issue
- **Documentation:** Check the rendered docs (Read the Docs)

Thank you for contributing to diffpes and advancing differentiable ARPES
simulation!
