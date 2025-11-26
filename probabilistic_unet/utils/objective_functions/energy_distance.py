def get_energy_distance_components(self, gt_seg_modes, seg_samples, eval_class_ids, ignore_mask=None):
    """
    Calculates the components for the IoU-based generalized energy distance given an array holding all segmentation
    modes and an array holding all sampled segmentations.
    :param gt_seg_modes: N-D array in format (num_modes,[...],H,W)
    :param seg_samples: N-D array in format (num_samples,[...],H,W)
    :param eval_class_ids: integer or list of integers specifying the classes to encode, if integer range() is applied
    :param ignore_mask: N-D array in format ([...],H,W)
    :return: dict
    """
    num_modes = gt_seg_modes.shape[0]
    num_samples = seg_samples.shape[0]

    if isinstance(eval_class_ids, int):
        eval_class_ids = list(range(eval_class_ids))

    d_matrix_YS = np.zeros(shape=(num_modes, num_samples, len(eval_class_ids)), dtype=np.float32)
    d_matrix_YY = np.zeros(shape=(num_modes, num_modes, len(eval_class_ids)), dtype=np.float32)
    d_matrix_SS = np.zeros(shape=(num_samples, num_samples, len(eval_class_ids)), dtype=np.float32)

    # iterate all ground-truth modes
    for mode in range(num_modes):

        ##########################################
        #   Calculate d(Y,S) = [1 - IoU(Y,S)],	 #
        #   with S ~ P_pred, Y ~ P_gt  			 #
        ##########################################

        # iterate the samples S
        for i in range(num_samples):
            conf_matrix = training_utils.calc_confusion(gt_seg_modes[mode], seg_samples[i],
                                                        loss_mask=ignore_mask, class_ixs=eval_class_ids)
            iou = training_utils.metrics_from_conf_matrix(conf_matrix)['iou']
            d_matrix_YS[mode, i] = 1. - iou

        ###########################################
        #   Calculate d(Y,Y') = [1 - IoU(Y,Y')],  #
        #   with Y,Y' ~ P_gt  	   				  #
        ###########################################

        # iterate the ground-truth modes Y' while exploiting the pair-wise symmetries for efficiency
        for mode_2 in range(mode, num_modes):
            conf_matrix = training_utils.calc_confusion(gt_seg_modes[mode], gt_seg_modes[mode_2],
                                                        loss_mask=ignore_mask, class_ixs=eval_class_ids)
            iou = training_utils.metrics_from_conf_matrix(conf_matrix)['iou']
            d_matrix_YY[mode, mode_2] = 1. - iou
            d_matrix_YY[mode_2, mode] = 1. - iou

    #########################################
    #   Calculate d(S,S') = 1 - IoU(S,S'),  #
    #   with S,S' ~ P_pred        			#
    #########################################

    # iterate all samples S
    for i in range(num_samples):
        # iterate all samples S'
        for j in range(i, num_samples):
            conf_matrix = training_utils.calc_confusion(seg_samples[i], seg_samples[j],
                                                        loss_mask=ignore_mask, class_ixs=eval_class_ids)
            iou = training_utils.metrics_from_conf_matrix(conf_matrix)['iou']
            d_matrix_SS[i, j] = 1. - iou
            d_matrix_SS[j, i] = 1. - iou

    return {'YS': d_matrix_YS, 'SS': d_matrix_SS, 'YY': d_matrix_YY}