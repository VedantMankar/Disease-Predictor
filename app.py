from flask import Flask , request,render_template
import os
import pickle
import h5py
import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
from PIL import Image 


#loading models
diabetes_model = pickle.load(open("diabetes_model.pkl","rb"))
breast_cancer_model = pickle.load(open("breastcancer_model.pkl","rb"))
heart_attack_model = pickle.load(open("heart_model.pkl","rb"))
malaria_model = load_model("cnn_model.h5")


APP_ROOT = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "C:/Users/Vedant/Desktop/coding/PYTHON/Python Projects/DiseasePredictor/static/"
app = Flask(__name__)
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/Diabetes',methods=['GET','POST'])
def diabetes():
    if request.method == "POST":
        #Pregnancy
        pregnancies = int(request.form['Pregnancies'])
        #Glucose
        Glucose = int(request.form['Glucose'])
        #BloodPressure
        BloodPressure = int(request.form['BloodPressure'])
        #SkinThickness
        SkinThickness = int(request.form['SkinThickness'])
        #Insulin
        Insulin = int(request.form['Insulin'])
        #BMI
        BMI = float(request.form['BMI'])
        #DiabetesPedigreeFunction
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        #Age
        Age = int(request.form['Age'])
        lst = np.array([pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        lst = np.reshape(lst,(-1,1))
        x = st.fit_transform(lst)
        x = np.reshape(x,(8,))
        print(x)
        
        output = diabetes_model.predict([x])
        output = int(output[0])
        print(output)
        if output == 1:
            return render_template("diabetes_html.html",prediction_text="You have a possibility of Diabetes")
        else:
            return render_template("diabetes_html.html",prediction_text="Yayy !!!! You Do not have any disease")
    else:
        return render_template("diabetes_html.html")

@app.route("/Breast_cancer",methods=['POST','GET'])
def breastcancer():
    if request.method == "POST":
        #Mean Radius
        mean_radius = float(request.form['mean_radius'])
        #Mean Texture
        mean_texture = float(request.form['mean_texture'])
        #Mean Perimeter
        mean_perimeter = float(request.form['mean_perimeter'])
        #Mean Area
        mean_area = float(request.form['mean_area'])
        #Mean Smoothness
        mean_smoothness = float(request.form['mean_smoothness'])

        lst = np.array([[mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness]])
        lst = np.reshape(lst,(-1,1))
        x = st.fit(lst).transform(lst)
        x = np.reshape(x,(1,5))
        output = breast_cancer_model.predict(x)
        output = output[0]
        print(output)
        if output == 1:
            return render_template("breast_cancer.html",prediction_text="You have a Possibility of Breast Cancer!!!")
        else:
            return render_template("breast_cancer.html",prediction_text="Yayy!!! you do not have any Disease")
    else:
        return render_template("breast_cancer.html")

@app.route('/Heart',methods=['POST','GET'])
def heart_attack():
    if request.method == 'POST':
        age = int(request.form['age'])
        trestbps = int(request.form['trestbps'])
        cholestrol = int(request.form['cholestrol'])
        max_heart_rate_achieved = int(request.form['thalach'])
        st_depression = float(request.form['Oldpeak'])
        num_major_vessels = int(request.form['ca'])
        sex = request.form['Sex']
        if sex == "Male":
            sex_male=1
        else:
            sex_male=0
        chest_pain = request.form['cp']
        if chest_pain == "typical_angina":
            chest_pain_type_atypical_angina = 0
            chest_pain_type_non_anginal_pain = 0
            chest_pain_type_typical_angina = 1
        elif chest_pain == "atypical_angina":
            chest_pain_type_atypical_angina = 1
            chest_pain_type_non_anginal_pain = 0
            chest_pain_type_typical_angina = 0
        elif chest_pain == "non_anginal_pain":
            chest_pain_type_atypical_angina = 0
            chest_pain_type_non_anginal_pain = 1
            chest_pain_type_typical_angina = 0
        else:
            chest_pain_type_typical_angina = 0
            chest_pain_type_non_anginal_pain = 0
            chest_pain_type_atypical_angina = 0
        fasting_bs = request.form['fbs']
        if fasting_bs == "greater_than_120":
            fasting_blood_sugar_lower_than_120 = 0
        else:
            fasting_blood_sugar_lower_than_120 = 1
        restecg = request.form['restecg']
        if restecg == "Normal":
            rest_ecg_normal = 1
            rest_left_vhp = 0
        elif restecg == "Left Ventricular Hypertrophy":
            rest_left_vhp = 1
            rest_ecg_normal = 0
        else:
            rest_ecg_normal = 0
            rest_left_vhp = 0
        Exang = request.form['exang']
        if Exang == "Yes":
            exercise_induced_angina_yes = 1
        else:
            exercise_induced_angina_yes = 0
        slope = request.form['slope']
        if slope == "upsloping":
            st_slope_flat = 0
            st_slope_upsloping = 1
        elif slope == "flat":
            st_slope_flat = 1
            st_slope_upsloping = 0
        else:
            st_slope_flat = 0
            st_slope_upsloping = 0
        thal = request.form['thal']
        if thal == "Fixed_Defect":
            thalassemia_fixed_defect = 1
            thalassemia_normal = 0
            thalassemia_reversable_defect = 0
        elif thal == "Normal":
            thalassemia_fixed_defect = 0
            thalassemia_normal = 1
            thalassemia_reversable_defect =0
        else:
            thalassemia_fixed_defect = 0
            thalassemia_normal = 0
            thalassemia_reversable_defect = 1

        features = np.array([age,trestbps,cholestrol,max_heart_rate_achieved,st_depression,num_major_vessels,
        sex_male,chest_pain_type_atypical_angina,chest_pain_type_non_anginal_pain,chest_pain_type_typical_angina,
        fasting_blood_sugar_lower_than_120,rest_left_vhp,rest_ecg_normal,exercise_induced_angina_yes,st_slope_flat,
        st_slope_upsloping,thalassemia_fixed_defect,thalassemia_normal,thalassemia_reversable_defect])
        x = np.reshape(features,(1,19))
        output = heart_attack_model.predict(x)
        output = output[0]
        if output == 1:
            return render_template("heart_attack.html",prediction_text="You should Probably visit a cardiologist")
        else:
            return render_template("heart_attack.html",prediction_text="Yayy!! you do not have any problem")
    return render_template("heart_attack.html")



@app.route('/Malaria',methods=['GET','POST'])
def malaria():
    if request.method == "POST":
        if request.files:
            img = request.files['image']
            full_path  = os.path.join(UPLOAD_FOLDER,img.filename)
            img.save(full_path)
            data = load_img(full_path)
            input_arr = img_to_array(data)
            input_arr = np.resize(input_arr,(128,128,3))
            input_arr = np.array([input_arr])
            pred = malaria_model.predict_classes(input_arr)
            output = pred[0]
            if output == 0:
                return render_template("predict.html",filename=img.filename,prediction_text="This cell is Parasitized")
            else:
                return render_template("predict.html",filename=img.filename,prediction_text="This cell is Uninfected")
        
    else:
        return render_template("malaria.html")



if __name__ == "__main__":
    app.run(debug=True,threaded=False)