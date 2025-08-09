"""
Zero-shot configuration: Ark pretrained tasks, dataset label aliases, and optional prompts.
"""
from __future__ import annotations
from typing import Dict, List

# Ark pretrained task mapping 
ARK_PRETRAINED_TASKS: Dict[str, Dict] = {
    "MIMIC": {
        "num_classes": 14,
        "head_index": 0,
        "task_type": "multilabel",
        "diseases": [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
            'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
    },
    "CheXpert": {
        "num_classes": 14,
        "head_index": 1,
        "task_type": "multilabel",
        "diseases": [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
            'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
    },
    "ChestXray14": {
        "num_classes": 14,
        "head_index": 2,
        "task_type": "multilabel",
        "diseases": [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
            'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural thickening', 'Hernia'
        ]
    },
    "RSNA-Pneumonia": {
        "num_classes": 3,
        "head_index": 3,
        "task_type": "multiclass",
        "diseases": ['No Lung Opacity/Not Normal', 'Normal', 'Lung Opacity']
    },
    "VinDr-CXR": {
        "num_classes": 6,
        "head_index": 4,
        "task_type": "multilabel",
        "diseases": ['Pleural effusion', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding']
    },
    "Shenzhen": {
        "num_classes": 1,
        "head_index": 5,
        "task_type": "binary",
        "diseases": ['Tuberculosis']
    }
}

#  Dictionary mapping between downstream dataset labels (keys) and pretrained Ark class label aliases (values)
DATASET_LABEL_ALIASES: Dict[str, Dict[str, List[str]]] = {
    "VinDr-CXR": {
        "no finding": ["no finding"],
        "cardiomegaly": ["cardiomegaly"],
        "pleural effusion": ["effusion", "pleural effusion"],
        "pleural thickening": ["pleural thickening"],
        "lung opacity": ["lung opacity"],
        "pulmonary fibrosis": ["pulmonary fibrosis"],
        "tuberculosis": ["tuberculosis"],
        "aortic enlargement": ["enlarged cardiomediastinum"],
    },
    "RSNA-Pneumonia": {
        "pneumonia": ["pneumonia"]
    }
}




