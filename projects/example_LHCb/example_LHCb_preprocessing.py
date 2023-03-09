import uproot
import numpy as np

# Open pre-Baler data as pandas df
infile = uproot.open("./data/example_LHCb/example_LHCb.root")["DecayTree"]
df = infile.arrays(infile.keys(), library="pd")

# Perform PID cuts so we are looking at B->KKK
df = df[(df.H1_ProbK > 0.5) & (df.H2_ProbK > 0.5) & (df.H3_ProbK > 0.5)].copy()

# Export as pkl file for input to Baler
names = np.array(list(df.columns))
data = df.to_numpy()

np.save("./data/example_LHCb/example_LHCb_names.npy", names)
np.save("./data/example_LHCb/example_LHCb_data.npy", data)
