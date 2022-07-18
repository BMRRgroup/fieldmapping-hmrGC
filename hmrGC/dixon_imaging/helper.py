import numpy as np


def calculate_pdsf_percent(W, F, S):
    WFS = np.abs(W + F + S)
    wfs = np.abs(W) + np.abs(F) + np.abs(S)
    Wr = np.divide(np.abs(W + F), WFS, out=np.zeros_like(WFS), where=WFS!=0)
    Sr = np.divide(np.abs(S), WFS, out=np.zeros_like(WFS), where=WFS!=0)
    pdsf_abs = 100 * np.divide(np.abs(S), wfs, out=np.zeros_like(wfs), where=wfs!=0)
    return 100 * np.where((0 < pdsf_abs) & (pdsf_abs <= 50), (1 -Wr), Sr)

def calculate_pdff_percent(W, F, S=None):
    if S is None:
        WF = np.abs(W + F)
        wf = np.abs(W) + np.abs(F)
        Wr = np.divide(np.abs(W), WF, out=np.zeros_like(WF), where=WF!=0)
    else:
        WF = np.abs(W + F + S)
        wf = np.abs(W) + np.abs(F) + np.abs(S)
        WS = np.abs(W + S)
        Wr = np.divide(WS, WF, out=np.zeros_like(WF), where=WF!=0)
    Fr = np.divide(np.abs(F), WF, out=np.zeros_like(WF), where=WF!=0)
    pdff_abs = 100 * np.divide(np.abs(F), wf, out=np.zeros_like(wf), where=wf!=0)
    return 100 * np.where((0 < pdff_abs) & (pdff_abs <= 50), (1 - Wr), Fr)
