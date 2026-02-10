import pickle
user_encoder, _ = pickle.load(open("encoders.pkl","rb"))
print(user_encoder.classes_)
