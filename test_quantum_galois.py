"""
Unit tests for Quantum Galois Theory module.

Tests for the quantum algorithms implemented for Galois theory 
and computational algebra problems.

Author: Aaron Franklin
"""

import unittest
import numpy as np
import sympy as sp
from sympy.combinatorics.perm_groups import PermutationGroup
from quantum_galois import (
    QuantumGaloisGroupSolver,
    QuantumPolynomialFactorizer,
    QuantumRootFinder,
    QuantumFieldExtensionCalculator
)


class TestQuantumGaloisGroupSolver(unittest.TestCase):
    """Tests for the QuantumGaloisGroupSolver class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.solver = QuantumGaloisGroupSolver()
        self.x = sp.symbols('x')
    
    def test_quadratic_polynomial(self):
        """Test Galois group computation for a quadratic polynomial."""
        poly = sp.Poly(self.x**2 - 2, self.x)
        group = self.solver.compute_galois_group(poly)
        
        # Should be Z2 (cyclic group of order 2)
        self.assertEqual(group.order(), 2)
        
        # Check if the group contains the identity and a transposition
        elements = list(group.elements)
        self.assertTrue(len(elements) == 2)
    
    def test_cubic_polynomial(self):
        """Test Galois group computation for a cubic polynomial."""
        poly = sp.Poly(self.x**3 - 3*self.x - 1, self.x)
        group = self.solver.compute_galois_group(poly)
        
        # Should be S3 (symmetric group on 3 elements)
        self.assertEqual(group.order(), 6)
        
        # Check if the group contains the expected number of elements
        elements = list(group.elements)
        self.assertTrue(len(elements) == 6)
    
    def test_discriminant_computation(self):
        """Test discriminant computation."""
        poly = sp.Poly(self.x**2 - 2, self.x)
        discriminant = self.solver._compute_discriminant(poly)
        
        # Discriminant of x^2 - 2 is 8
        self.assertEqual(discriminant, 8)


class TestQuantumPolynomialFactorizer(unittest.TestCase):
    """Tests for the QuantumPolynomialFactorizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factorizer = QuantumPolynomialFactorizer()
        self.x = sp.symbols('x')
    
    def test_irreducible_polynomial(self):
        """Test factorization of an irreducible polynomial over a finite field."""
        poly = sp.Poly(self.x**2 + 1, self.x)
        field_prime = 7
        
        factors = self.factorizer.factor_polynomial(poly, field_prime)
        
        # x^2 + 1 is irreducible over GF(7)
        self.assertEqual(len(factors), 1)
        self.assertEqual(factors[0][1], 1)  # Multiplicity should be 1
    
    def test_reducible_polynomial(self):
        """Test factorization of a reducible polynomial over a finite field."""
        poly = sp.Poly(self.x**2 - 1, self.x)
        field_prime = 7
        
        factors = self.factorizer.factor_polynomial(poly, field_prime)
        
        # x^2 - 1 = (x-1)(x+1) over any field of characteristic != 2
        self.assertEqual(len(factors), 2)
        
        # Check that both factors have multiplicity 1
        self.assertEqual(factors[0][1], 1)
        self.assertEqual(factors[1][1], 1)
    
    def test_polynomial_gcd(self):
        """Test GCD computation for polynomials over a finite field."""
        poly1 = sp.Poly(self.x**2 - 1, self.x)
        poly2 = sp.Poly(self.x - 1, self.x)
        field_prime = 7
        
        gcd = self.factorizer.quantum_polynomial_gcd(poly1, poly2, field_prime)
        
        # GCD of (x^2 - 1) and (x - 1) should be (x - 1)
        self.assertEqual(gcd.degree(), 1)
        self.assertEqual(gcd.LC(), 1)
        self.assertEqual(gcd.eval(1), 0)


class TestQuantumRootFinder(unittest.TestCase):
    """Tests for the QuantumRootFinder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.root_finder = QuantumRootFinder()
        self.x = sp.symbols('x')
    
    def test_quadratic_roots(self):
        """Test finding roots of a quadratic polynomial."""
        poly = sp.Poly(self.x**2 - 2, self.x)
        roots = self.root_finder.find_roots(poly)
        
        # Roots should be +sqrt(2) and -sqrt(2)
        self.assertEqual(len(roots), 2)
        
        # Check that the roots are correct
        self.assertAlmostEqual(abs(roots[0]), np.sqrt(2), places=10)
        self.assertAlmostEqual(abs(roots[1]), np.sqrt(2), places=10)
        self.assertAlmostEqual(roots[0] * roots[1], -2, places=10)
    
    def test_roots_in_finite_field(self):
        """Test finding roots of a polynomial over a finite field."""
        poly = sp.Poly(self.x**2 - 1, self.x)
        field_prime = 7
        
        roots = self.root_finder.quantum_find_roots_in_finite_field(poly, field_prime)
        
        # x^2 - 1 = 0 has two solutions in GF(7): 1 and 6 (= -1 mod 7)
        self.assertEqual(len(roots), 2)
        self.assertTrue(1 in roots)
        self.assertTrue(6 in roots)


class TestQuantumFieldExtensionCalculator(unittest.TestCase):
    """Tests for the QuantumFieldExtensionCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = QuantumFieldExtensionCalculator()
        self.x = sp.symbols('x')
    
    def test_splitting_field(self):
        """Test computation of splitting field."""
        poly = sp.Poly(self.x**2 - 2, self.x)
        minimal_poly, primitive_elements = self.calculator.compute_splitting_field(poly)
        
        # Should return a minimal polynomial and at least one primitive element
        self.assertIsInstance(minimal_poly, sp.Poly)
        self.assertTrue(len(primitive_elements) > 0)
    
    def test_automorphisms(self):
        """Test computation of automorphisms of the splitting field."""
        poly = sp.Poly(self.x**2 - 2, self.x)
        automorphisms = self.calculator.compute_automorphisms(poly)
        
        # For x^2 - 2, there should be
