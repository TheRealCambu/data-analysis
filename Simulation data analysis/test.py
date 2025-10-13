import numpy as np

data_path = (r"C:\Users\39338\Politecnico Di Torino Studenti Dropbox\Simone Cambursano\Politecnico"
             r"\Tesi\Data-analysis\Simulation Sweeps\Rate Sweeps\Second Batch\results_QPSK_40.0GBaud_CRAlgo_fd.npz")

data = dict(np.load(data_path, allow_pickle=True))

for key, value in data.items():
    print(f"Key: {key} - Value: {value}")
