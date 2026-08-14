Testing & Validation
====================

This section publishes diffpes's test suite as a validation reference: what
the library guarantees, where each guarantee is checked, and how the check is
implemented. The pages mirror ``tests/test_diffpes`` by public module and
subpackage. Thus, the validation surface lines up with the public API surface.

How to Read
-----------

Each module page starts with the test file's coverage scope. Each ``Test*``
class groups checks for a source symbol or behavior family, and each
``test_*`` method documents the property being verified along with the
fixtures, parametrization, assertions, and JAX execution path used to
verify it.

Bidirectional Links
-------------------

Source docstrings may point forward to the tests that validate them via
``:see:`` references, and test classes point back to the source symbols
they cover, so the API reference and this validation reference stay
navigable in both directions.

Coverage Map
------------

.. toctree::
   :maxdepth: 1

   certify
   constants
   contracts
   inout
   maths
   matrixel
   plots
   radial
   simul
   tightb
   types
   gradients
