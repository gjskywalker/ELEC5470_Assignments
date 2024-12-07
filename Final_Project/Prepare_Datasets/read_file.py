import pickle

with open("out_of_sample_indicators.pkl", "rb") as f:
    out_of_sample_indicators = pickle.load(f)
    for i in out_of_sample_indicators:
        print(out_of_sample_indicators_array[i]['train'])
        print(out_of_sample_indicators_array[i]['test'])
    print(out_of_sample_indicators)

# with open("out_of_sample_indicator_array.pkl", "rb") as f:
#     out_of_sample_indicator_array = pickle.load(f)
#     print(out_of_sample_indicator_array)    

# with open("out_of_sample_labels.pkl", "rb") as f:
#     out_of_sample_labels = pickle.load(f)
#     print(out_of_sample_labels)