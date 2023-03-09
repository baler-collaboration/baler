import uproot

#Open pre-Baler data as pandas df
tuple_pre=uproot.open("./data/example_LHCb/example_LHCb.root")["DecayTree"]
df=tuple_pre.arrays(tuple_pre.keys(), library='pd')

#Perform PID cuts so we are looking at B->KKK
df_pre=df[(df.H1_ProbK>0.5) & (df.H2_ProbK>0.5) & (df.H3_ProbK>0.5)].copy()

#Export as pkl file for input to Baler
df_pre.to_pickle("./data/example_LHCb/example_LHCb.pickle")
