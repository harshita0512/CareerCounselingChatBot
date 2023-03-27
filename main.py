import pickle
# import numpy as np
# from sklearn import *
# from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='Template')

enc = pickle.load(open('enc.pkl', 'rb'))


@app.route('/')
def hello():
    return render_template("index.html")


@app.route("/pre", methods=['POST'])
def pre():
    programming = request.form.get('Programming')
    cn = request.form.get('ComputerNetworks')
    lq = request.form.get('logical_quotient')
    coding = request.form.get('coding_skills')
    ps = request.form.get('public_speaking')
    self_learn = request.form.get('Self_learn')
    cs = request.form.get('Certifications')
    workshop = request.form.get('Workshops')
    memory = request.form.get('Memory')
    interested_sub = request.form.get('InterestedSubjects')
    interested_area = request.form.get('InterestedCareerarea')
    company_type = request.form.get('TypeofCompany')
    job_role = request.form.get('jobRole')
    work_team = request.form.get('work_teams')

    input_features = pd.DataFrame(
        {'Programming': [programming], 'ComputerNetworks': [cn], 'logical_quotient': [lq], 'coding_skills': [coding],
         'public_speaking': [ps], 'Self_learn': [self_learn], 'Certifications': [cs], 'Workshops': [workshop]
            , 'Memory': [memory], 'InteretedSubjects': [interested_sub], 'InterestedCareerarea': [interested_area],
         'TypeofCompany': [company_type], 'jobRole': [job_role], 'work_teams': [work_team]})

    # One-hot encode the input features
    input_features = enc.transform(input_features).toarray()

    result = model.predict(input_features)

    return render_template('index.html', result=str(result))


# return jsonify({'Suggested Job Role': result.tolist()})


if __name__ == '__main__':
    model = pickle.load(open('model.pkl', 'rb'))
    app.run(debug=True)
