.. _rotmat-label:

Rotation Matrices
=================

Definition of rotation matrices
-------------------------------

* :func:`rotmat.R` ... 3D rotation matrix for rotation about a coordinate axis

Conversion Routines
-------------------
* :func:`rotmat.convert` ... Convert a rotation matrix to the corresponding quaternion
* :func:`rotmat.sequence` ... Calculation of Euler, Fick/aeronautic, Helmholtz angles
* :func:`rotmat.seq2quat` ... Calculation of quaternions from Euler, Fick/aeronautic, Helmholtz angles

Symbolic matrices
-----------------

* :func:`rotmat.R_s()` ... symbolix matrix for rotation about a coordinate axis

For example, you can e.g. generate a Fick-matrix, with

    R_Fick = R_s(2, 'theta') * R_s(1, 'phi') * R_s(0, 'psi')

*Note:* For displaying the Greek symbols for LaTeX expressions, such as 'psi', you may have to add the following lines to your Jupyter terminal:

>>> from sympy.interactive import printing
>>> printing.init_printing(use_latex=True)


.. toctree::
   :maxdepth: 2

Spatial Transformation Matrices
-------------------------------

* :func:`rotmat.stm` ... spatial transformation matrix, for combined rotations/translations
* :func:`rotmat.stm_s()` ... symbolix spatial transformation matrix

Denavit-Hartenberg Transformations
----------------------------------

* :func:`rotmat.dh` ... Denavit-Hartenberg transformation matrix
* :func:`rotmat.dh_s` ... symbolic Denavit-Hartenberg transformation matrix

Details
-------

.. automodule:: rotmat
    :members:
