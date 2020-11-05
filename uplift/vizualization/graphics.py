import matplotlib.pyplot as plt
import numpy as np
import pylab

  
from sklearn.utils.validation import check_consistent_length
from ..metrics import qini_curve, perfect_qini_curve, qini_auc_score


def plot_qini_curve(y_reactions, uplift_score, treatment, random=True, perfect=True):
    """Plot Qini curves from predictions."""
   
    check_consistent_length(y_reactions, uplift_score, treatment)
    y_reactions, uplift_score, treatment = np.array(y_reactions), np.array(uplift_score), np.array(treatment)

   

    x_actual, y_actual = qini_curve(y_reactions, uplift_score, treatment)

    pylab.plot(x_actual, y_actual, label='Our model', color='green')
    if random:
        x_baseline, y_baseline = x_actual, x_actual * y_actual[-1] / len(y_reactions)
        pylab.plot(x_baseline, y_baseline, label='Random model', color='black')
        

    if perfect:
        x_perfect, y_perfect = perfect_qini_curve(y_reactions, treatment)
        pylab.plot(x_perfect, y_perfect, label='Perfect model', color='Red')
        
    pylab.fill_between(x_actual,y_baseline,y_actual,color="red")
    pylab.grid(True)
    pylab.xlabel('Treat num')
    pylab.ylabel('Uplift reactions')
    pylab.title('Qini curve')
    pylab.legend(loc='lower right')
    return pylab


