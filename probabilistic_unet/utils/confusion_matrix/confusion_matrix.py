def multiclass_iou(output, target, classes):
    
    smooth = 1e-6
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    iou_scores = {}
    for cls, className in enumerate(classes):
        
        output_cls = torch.where(output == cls, 1, 0)
        target_cls = torch.where(target == cls, 1, 0)

        intersection = (output_cls * target_cls).sum()
        union = (output_cls + target_cls).sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_scores[className] = iou.nanmean()
        
        
    mean_iou = torch.stack(list(iou_scores.values())).nanmean()
    return mean_iou, iou_scores
    
def SingleImageConfusionMatrix(Predictions, Lables, classes):
    
#     smooth = 1e-6
    CM_unnormalised = torch.zeros([8,8]) 
    total_pixels_in_class_j = torch.zeros([8,1])
    Predictions = torch.argmax(Predictions, dim=0)
    Lables = torch.argmax(Lables, dim=0)

    # produce unnormalised CM:
    for j,_ in enumerate(classes):
        is_pixel_in_class_j = torch.where(Lables == j, 1, 0)
        total_pixels_in_class_j[j] = is_pixel_in_class_j.sum()

        for i,_ in enumerate(classes):

            is_pixel_predicted_as_i = torch.where(Predictions == i, 1, 0)
            is_pixel_predicted_as_i_and_is_in_class_j = is_pixel_in_class_j * is_pixel_predicted_as_i
            total_pixels_predicted_as_i_in_class_j = is_pixel_predicted_as_i_and_is_in_class_j.sum()
            CM_unnormalised[i][j] = total_pixels_predicted_as_i_in_class_j

    return np.array([CM_unnormalised, total_pixels_in_class_j])


def BatchImageConfusionMatrix(predictions, labels, classes):
    
    smooth = 1e-3
    smooth = 0.
    pixels_in_class_j_totals = torch.zeros([8,1])
    CM_unnormalised_totals = torch.zeros([8,8])

    confusionTest = [SingleImageConfusionMatrix(pred, sample, classes) for pred, sample in zip(predictions, labels)]
    
    CM_unnormalised_totals = np.nan_to_num(np.array(list(zip(*confusionTest))[0])).sum()
    pixels_in_class_j_totals = np.nan_to_num(np.array(list(zip(*confusionTest))[1])).sum()
                                             
                                             
                                             
                                             
    CM = torch.tensor([100 * ((CM_unnormalised_totals[i][j] + smooth) / (pixels_in_class_j_totals[j] + smooth)) for i,_ in enumerate(classes) for j,_ in enumerate(classes)]).reshape(8,8)
    
    return CM
    
