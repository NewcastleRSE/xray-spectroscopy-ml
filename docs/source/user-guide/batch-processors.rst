Batch processors
================

A batch processor adapts a dataset batch to a model interface. 
XANESNET selects one automatically from the pair ``(dataset_type, model_type)`` using ``BatchProcessorRegistry``. 
Users normally do not configure one directly.

Available implementations
--------------------------

Forward processors:

* ``descriptor`` + ``mlp`` (:class:`~xanesnet.batchprocessors.forward.descriptor_mlp.DescriptorMLPBatchProcessor`)
* ``descriptor_multihead`` + ``mh_mlp`` or ``mh_cnn`` (:class:`~xanesnet.batchprocessors.forward.descriptor_multihead.DescriptorMultiheadBatchProcessor`)
* ``geometrygraph`` + ``schnet`` (:class:`~xanesnet.batchprocessors.forward.geometrygraph_schnet.GeometryGraphSchNetBatchProcessor`)
* ``geometrygraph`` + ``dimenet`` or ``dimenet++`` (:class:`~xanesnet.batchprocessors.forward.geometrygraph_dimenet.GeometryGraphDimeNetBatchProcessor`)
* ``gemnet`` + ``gemnet`` (:class:`~xanesnet.batchprocessors.forward.gemnet.GemNetBatchProcessor`)
* ``gemnet_oc`` + ``gemnet_oc`` (:class:`~xanesnet.batchprocessors.forward.gemnet_oc.GemNetOCBatchProcessor`)
* ``e3ee`` + ``e3ee`` (:class:`~xanesnet.batchprocessors.forward.e3ee.E3EEBatchProcessor`)
* ``e3ee_full`` + ``e3ee_full`` (:class:`~xanesnet.batchprocessors.forward.e3ee_full.E3EEFullBatchProcessor`)
* ``envembed`` + ``envembed`` (:class:`~xanesnet.batchprocessors.forward.envembed.EnvEmbedBatchProcessor`)

Inverse processors:

* ``descriptor_inverse`` + ``mlp`` (:class:`~xanesnet.batchprocessors.inverse.descriptor_mlp.InverseDescriptorMLPBatchProcessor`)
* ``descriptor_multihead_inverse`` + ``mh_mlp`` or ``mh_cnn`` (:class:`~xanesnet.batchprocessors.inverse.descriptor_multihead.InverseDescriptorMultiheadBatchProcessor`)

The corresponding ``_mp`` dataset variants use the same processors as their non-multiprocessing dataset types.
See :doc:`datasets` for the complete dataset-model compatibility matrix and :doc:`models` for model choices.

Required interface
------------------

Every concrete ``BatchProcessor`` must implement these abstract methods:

* ``input_preparation(batch)`` converts a collated dataset batch into the inputs expected by ``model.forward``.
* ``target_preparation(batch)`` extracts the raw target tensor used for loss calculation and prediction records.
* ``encode_input(inputs, elements)`` maps raw inputs into model input space.
* ``encode_target(targets, elements)`` maps raw targets into model prediction space before the loss is computed.
* ``decode_target(predictions, elements)`` maps model predictions back to raw target space after inference.
* ``sample_id_preparation(batch)`` returns the sample identifiers used in prediction output.

The base class also provides optional hooks for prediction reshaping, target-site elements, and target-site indices. 
Direction-specific base classes implement the encoding behavior: forward processors encode targets and decode spectral predictions,
while inverse processors encode spectral inputs and keep descriptor targets unchanged.

API reference
-------------

See :class:`~xanesnet.batchprocessors.base.BatchProcessor`, :class:`~xanesnet.batchprocessors.forward.base.ForwardBatchProcessor`,
:class:`~xanesnet.batchprocessors.inverse.base.InverseBatchProcessor`, and :mod:`xanesnet.batchprocessors` for the base interfaces, registry, and concrete implementations.
