==================================
Descriptor Type Parameter Examples
==================================



* **wacsf**

	.. code-block::

		descriptor:
		    type: wacsf
		    params:
		      r_min: 1.0
		      r_max: 6.0
		      n_g2: 16
		      n_g4: 32

* **rdc**

	.. code-block::

		descriptor:
		    type: rdc
		    params:
		      r_min: 0.0
		      r_max: 8.0
		      dr: 0.01
		      alpha: 10.0
		      use_charge = False
		      use_spin: False

* **soap**

	.. code-block::

		descriptor:
		    type: soap
		    params:
		       species: [Fe, H, C, O, N, F, P, S, Cl, Br, I, Si, B, Se, As]
		       n_max: 8
		       l_max: 6
		       r_cut: 6.0

* **mbtr**

	.. code-block::

		descriptor:
		    type: mbtr
		    params:
		       species: [Fe, H, C, O, N, F, P, S, Cl, Br, I, Si, B, Se, As]
		       k2:
		         geometry:
		            function: distance
		         grid:
		            min: 0
		            max: 6
		            n: 100
		            sigma: 0.1
		         weighting:
		            function: exp
		            scale: 0.5
		            threshold: 0.001
		       k3:
		         geometry:
		            function: angle
		         grid:
		            min: 0
		            max: 180
		            n: 100
		            sigma: 0.1
		         weighting:
		            function: exp
		            scale: 0.5
		            threshold: 0.001
		       periodic: False
		       normalization: l2_each

* **lmbtr**

	.. code-block::

		descriptor:
		    type: lmbtr
		    params:
		       species: [Fe, H, C, O, N, F, P, S, Cl, Br, I, Si, B, Se, As]
		       k2:
		         geometry:
		            function: distance
		         grid:
		            min: 0
		            max: 6
		            n: 100
		            sigma: 0.1
		         weighting:
		            function: exp
		            scale: 0.5
		            threshold: 0.001
		       k3:
		         geometry:
		            function: angle
		         grid:
		            min: 0
		            max: 180
		            n: 100
		            sigma: 0.1
		         weighting:
		            function: exp
		            scale: 0.5
		            threshold: 0.001
		       periodic: False
		       normalization: l2_each