#|| 
These are procedures to compute coefficients for multistep
integrators, to accompany the paper
TY  - JOUR
AU  - Berry, Matthew M.
AU  - Healy, Liam M.
JF  - The Journal of the Astronautical Sciences
PY  - 2004///July-September
VL  - 52
IS  - 3
SP  - 331
EP  - 357
TI  - Implementation of Gauss-Jackson Integration for Orbit Propagation
UR  - http://hdl.handle.net/1903/2202
ER  - 
Equation and table references here refer to this paper.

The definitions are written in ANSI Common Lisp
(http://www.lispworks.com/documentation/HyperSpec/Front/index.htm) and
should work with any compliant implementation.

;;; Compile and load
(compile-file "coefficients")
(load "coefficients")

;;; How coefficient sets in paper were generated:
(map0-n #'adams-moulton-corrector 8)	; (27) Adams-Moulton corrector
(map0-n (sum #'adams-moulton-corrector) 8) ; (32) Adams-Bashforth predictor
(map0-n #'cowell-corrector 8)		; (44) Cowell corrector
(map0-n (sum #'cowell-corrector) 8)	; (48) Stoermer predictor
(coefficient-table (shift #'adams-moulton-corrector) 8 nil 1) ; Table 3
(coefficient-table (shift #'cowell-corrector 2) 8) ; Table 4
(mapa-b ; (68) last row
 (difference-to-ordinate (shift #'adams-moulton-corrector) 4) 4 0 -1)
;; (70) last row
(let ((summed-adams-moulton-corrector (shift #'adams-moulton-corrector)))
  (mapa-b (change-acceleration (difference-to-ordinate summed-adams-moulton-corrector 4)
		      (funcall summed-adams-moulton-corrector 0)
		      4)
	  4 0 -1))
(half-acceleration-in-sum ; Table 5
 (coefficient-table (shift #'adams-moulton-corrector) 8 t 1))
(coefficient-table (shift #'cowell-corrector 2) 8 t) ; Table 6

;;; Liam Healy
;;; May 24, 2005.
||#

(in-package :cl-user)

;;; Generators are functions of a non-negative index that gives the
;;; coefficient for that index.  Transformations produce a generator
;;; from another generator.

;;;;****************************************************************************
;;;; Integrators
;;;;****************************************************************************

(defun adams-moulton-corrector (n)
  "A generator for non-summed Adams-Moulton corrector in difference form. 
   Generate (27) with (map0-n #'adams-moulton-corrector 8).
   This is a basic coefficient set from which
   all predictor-corrector coefficients are derived."
  (if (= n 0)
      1
    (loop for i from 0 below n
	sum (/ (- (adams-moulton-corrector i)) (+ n 1 (- i))))))

(defun half-acceleration-in-sum (table)
  "Alter the ordinate coefficient table so that the sum (del^-1) term only includes
   summation up to the point being corrected in the midcorrector,
   and only 1/2 of the current point.  See explanation on page 346."
  (loop for row in table for j from 0
     collect
     (loop for col in row for k from 0
	collect
	(if
	 (= k j) (+ 1/2 col)
	 col))))

(defun cowell-corrector (n)
  "A generator for the Cowell corrector.  The form (map0-n #'cowell-corrector 8)
   will generate (44)."
  (loop for k from 0 to n
     sum (* (adams-moulton-corrector k)
	    (adams-moulton-corrector (- n k)))))

;;;;****************************************************************************
;;;; Transformations
;;;;****************************************************************************

;;; Transformations take a generator and produce another generator.
;;; They are of general applicability to different integration methods.
;;; These definitions are transformations or assist in making tables
;;; and definitions.

(defun sum (function &optional (initial 0))
  "Transform a coefficient generator to make successive sums of
   coefficients.  Apply to a corrector generator to produce the
   predictor generator.  As an operator, apply (1-Del)^-1, or
   move down the chart on p. 342."
  (labels ((psum (i)
	     (cond ((minusp i) initial)
		   (t (+ (funcall function i) (psum (1- i)))))))
    #'psum))
  
(defun difference (function &optional (start 0))
  "Transform a coefficient generator to make successive differences of
   coefficients.  Apply to a predictor generator to produce the
   corrector generator, and to the corrector generator 
   to produce the mid-corrector generators.  As an operator, apply
   (1-Del), or move up the chart on p. 342."
  (lambda (i)
    (if (> i start)
	(- (funcall function i) (funcall function (- i 1)))
      (funcall function i))))

(defun shift (function &optional (shift 1))
  "Shift coefficients by the given amount.  This is used for computing
   the summed form of integrators, for example,
   (shift #'adams-moulton-corrector) is the generator for the
   summed Adams-Moulton corrector coefficients."
  (lambda (n) (funcall function (+ n shift))))

(defun difference-to-ordinate
    (difference-coefficient-function order)
  "Transform difference coefficient generator to ordinate coefficient
   generator.  Formula (67)."
  (lambda (k)
    (let ((m (- order k)))
      (* (expt -1 m)
	 (loop for i from m to order
	    sum (* (binomial i m)
		   (funcall difference-coefficient-function i)))))))

(defun change-acceleration (function value ordinate-position)
  "Remove the acceleration term."
  (lambda (n)
    (- (funcall function n)
       (if (= n ordinate-position) value 0))))

;;;;****************************************************************************
;;;; Making tables
;;;;****************************************************************************

(defun coefficient-table
    (corrector-function order &optional ordinate (initial 0))
  (flet ((ord (fn)
	   (if ordinate (difference-to-ordinate fn order) fn)))
    (reverse
     (cons
      (map0-n (ord (sum corrector-function initial)) order) ; predictor
      (loop for function = corrector-function then (difference function)
	 repeat (1+ order)
	 collect (map0-n (ord function) order))))))

;;;;****************************************************************************
;;;; Utility
;;;;****************************************************************************

(defun binomial (n m)
  "Calculate binomial coefficient (N,M) {N over M}."
  (let ((k (min m (- n m))))
    (cond ((minusp k) 0)
	  ((zerop  k) 1)
	  ((loop for i from 1 to k
		 for j = n then (1- j)
		 for b = n then (floor (* b j) i)
	       finally (return b))))))

;;; From Paul Graham
(defun map0-n (fn n)
  (declare (function fn) (fixnum n))
  "Apply the fn to the list of numbers 0...n."
  (mapa-b fn 0 n))

(defun mapa-b (fn a b &optional (step 1))
  (declare (function fn) (fixnum a b step))
  "Apply the fn to the list of numbers a...b, stepping with step."
  (do ((i a (+ i step))
       (result nil))
      ((funcall (if (plusp step) #'> #'<) i b) (nreverse result))
    (declare (fixnum i))
    (push (funcall fn i) result)))
