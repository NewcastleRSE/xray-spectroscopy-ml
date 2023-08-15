===============
X A N E S N E T
===============

.. image:: images/xanesnet_graphic.png
   :align: center

|

XANESNET is an open source project developed through the `Penfold Group <http://penfoldgroup.co.uk>`_ @ `Newcastle University <https://ncl.ac.uk>`_ which provides machine learning tools and techniques for the fast and automated analysis and prediction of X-ray spectroscopy data. This documentation provides a guide to interacting with and contributing to XANESNET. 

.. The XANESNET code addresses two fundamental challenges: the so-called forward (property/structure-to-spectrum) and reverse (spectrum-to-property/structure) mapping problems. The forward mapping appraoch is similar to the appraoch used by computational researchers in the sense that an input structure is used to generate a spectral observable. In this area the objective of XANESNET is to supplement and support analysis provided by first principles quantum mechnanical simulations. The reverse mapping problem is perhaps the more natural of the two, as it has a clear connection to the problem that X-ray spectroscopists face day-to-day in their work: how can a measurement/observable be interpreted? Here we are seeking to provide methodologies in allow the direct extraction of properties from a recorded spectrum.




.. toctree::
	:caption: Documentation

	install
	running
	models
	contributing

.. toctree::
	:maxdepth: 5
	:caption: Project Links
	:hidden:

	XANESNET GitHub Repository <https://github.com/NewcastleRSE/xray-spectroscopy-ml>
	XANESNET GitLab Repostiory <https://gitlab.com/team-xnet/xanesnet>
	XANESNET Training Datasets <https://gitlab.com/team-xnet/training-sets>
	Penfold Group <http://penfoldgroup.co.uk>
	Original Keras Implementation <https://gitlab.com/team-xnet/xanesnet_keras>

------------
Publications
------------

++++++++++++++
XANESNET Code:
++++++++++++++

`A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra <https://doi.org/10.1021/acs.jpca.0c03723>`_ - C. D. Rankine, M. M. M. Madkhali, and T. J. Penfold, *J. Phys. Chem. A*, 2020, **124**, 4263-4270.

`Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network <https://doi.org/10.1063/5.0087255>`_ - C. D. Rankine, and T. J. Penfold, *J. Chem. Phys.*, 2022, **156**, 164102.
 
++++++++++++++++++++++++++++
Extension to X-ray Emission:
++++++++++++++++++++++++++++

`A deep neural network for valence-to-core X-ray emission spectroscopy <https://doi.org/10.1080/00268976.2022.2123406>`_ - T. J. Penfold, and C. D. Rankine, *Mol. Phys.*, 2022, e2123406.


+++++++++++++++++
The Applications:
+++++++++++++++++

`On the Analysis of X-ray Absorption Spectra for Polyoxometallates <https://doi.org/10.1016/j.cplett.2021.138893>`_ - E. Falbo, C. D. Rankine, and T. J. Penfold, *Chem. Phys. Lett.*, 2021, **780**, 138893.

`Enhancing the Anaysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump-X-ray Probe Experiments <https://doi.org/10.1039/D0CP06244H>`_ - M. M. M. Madkhali, C. D. Rankine, and T. J. Penfold, *Phys. Chem. Chem. Phys.*, 2021, **23**, 9259-9269.

++++++++++++++
Miscellaneous:
++++++++++++++

`The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy <https://doi.org/10.3390/molecules25112715>`_ - M. M. M. Madkhali, C. D. Rankine, and T. J. Penfold, *Molecules*, 2020, **25**, 2715.


---------------
Citing XANESNET
---------------

.. code-block::

	@software{xanesnet,
	  author = {Penfold Group, Newcastle University},
	  title = {XANESNET},
	  url = {https://gitlab.com/team-xnet/xanesnet},
	  date = {2023},
	}









