import pandas as pd
import os
import numpy as np
import seaborn as sns
import datetime
import chardet
import time
from pandas.api.types import is_number
from matplotlib import pyplot as plt
from fpdf import FPDF

print("Generating results...")
results = pd.read_csv('soot-results.csv', sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)

loi = pd.read_csv('LOI.csv', sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)

result = {"True Negative":0, "False Negative":0, "True Positive":0, "False Positive":0}
for (LOIGroundTruth, confluence, lrOA, rlOA, lrPDG, rlPDG, lrDP, rlDF) in zip(loi["Locally Observable Interference"], results["Confluence Inter"], results["left right OA Inter"], results["right left OA Inter"], results["left right PDG-SDG"], results["right left PDG-SDG"], results["left right DFP-Inter"], results["right left DFP-Inter"]):
    rAux = ""
    AnalysesORResult = False
    if ("true" in confluence or "true" in lrOA or "true" in  rlOA or "true" in lrPDG or "true" in  rlPDG or "true" in lrDP or "true" in rlDF) :
        AnalysesORResult = True
    if (LOIGroundTruth == "No"):
        if (AnalysesORResult):
            rAux = "False Positive"
        else:
            rAux = "True Negative"
    if (LOIGroundTruth == "Yes"):
        if (AnalysesORResult):
            rAux = "True Positive"
        else:
            rAux = "False Negative"
    if (LOIGroundTruth != "-"):
        result[rAux] = result.get(rAux) + 1

print(result)
# Sensitivity: TPR = TP / (TP + FN)
# Precision: PPV = TP / (TP + FP)
# F1 Score: F1 = 2TP / (2TP + FP + FN)
# Accuracy: ACC = (TP + TN) / (P + N)

FP = result["False Positive"]
TP = result["True Positive"]
FN = result["False Negative"]
TN = result["True Negative"]

sensitivity = (TP / (TP + FN))
precision = (TP / (TP + FP))
f1_score = (2*TP / (2*TP + FP + FN))
accuracy = ((TP + TN) / (FP + TP + TN + FN))


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=16)
        plt.yticks(tick_marks, target_names, fontsize=16)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=23)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=23)
    plt.tight_layout()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig("confusion_matrix.jpg")
    plt.show()

plot_confusion_matrix(cm           = np.array([[TP,  FP],
                                               [FN, TN]]),
                      normalize    = False,
                      target_names = ['Positive', 'Negative'],
                      title        = "Confusion Matrix")
# variable pdf
pdf = FPDF()

# Add a page
pdf.add_page()

# set style and size of font
# that you want in the pdf
pdf.set_font("Arial", size = 15)

# create a cell
pdf.cell(200, 10, txt = "Results for execution",
         ln = 1, align = 'C')

pdf.cell(200, 10, txt = ("Precision: "+str(round(precision, 4))),
         ln = 2, align = 'L')

pdf.cell(200, 10, txt = ("Recall: "+str(round(sensitivity, 4))),
         ln = 2, align = 'L')

pdf.cell(200, 10, txt = ("F1 Score: "+str(round(f1_score, 4))),
         ln = 2, align = 'L')

pdf.cell(200, 10, txt = ("Accuracy: "+str(round(accuracy, 4))),
         ln = 2, align = 'L')

pdf.image("confusion_matrix.jpg", x = None, y = None, w = 160, h = 110, type = 'jpg', link = 'confusion_matrix.jpg')

# save the pdf with name .pdf
pdf.output("results.pdf")