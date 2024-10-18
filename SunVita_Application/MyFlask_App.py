from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root@localhost/flask'

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)


# Charger les données
data_path = 'Filtered_Patient.xlsx'
df = pd.read_excel(data_path)

# Reprendre le calcul des moyennes et des pivots comme prétraitement
# Calculer la moyenne des niveaux de vitamine D pour chaque IPP
vitamin_d_df = df[df['Examen'] == 'VITAMINE D']
vitamin_d_avg = vitamin_d_df.groupby('IPP')['Résultat'].mean().reset_index()
vitamin_d_avg = vitamin_d_avg.rename(columns={'Résultat': 'Vitamine_D'})

# Fusionner les données pour ajouter les moyennes de Vitamine D
data = df.merge(vitamin_d_avg, on='IPP')

# Exclure les entrées de Vitamine D des examens
data = data[data['Examen'] != 'VITAMINE D']

# Calculer la moyenne des résultats des examens par IPP et Examen
exam_avg = data.groupby(['IPP', 'Examen'])['Résultat'].mean().reset_index()

# Ajouter les moyennes de Vitamine D aux données
finaldata = exam_avg.merge(vitamin_d_avg, on='IPP')

# Ajouter l'âge et le sexe aux données finales
finaldata = finaldata.merge(df[['IPP', 'Sexe', 'AGE']].drop_duplicates(), on='IPP')

# Pivoter les données pour avoir chaque test comme une colonne avec les résultats sous chaque IPP
finaldata_pivot = finaldata.pivot(index='IPP', columns='Examen', values='Résultat').reset_index()

# Ajouter les colonnes d'âge, de sexe et de Vitamine D
finaldata_pivot = finaldata_pivot.merge(finaldata[['IPP', 'AGE', 'Sexe', 'Vitamine_D']].drop_duplicates(), on='IPP', how='left')

# Filtrer pour garder seulement les tests requis
tests = ['AGE', 'Sexe', 'CALCIUM', 'ALBUMINE', 'CRP', 'PROTÉINES TOTALES', 'LDH', 'GLUCOSE', 'SODIUM SANGUIN']
finaldata_pivot = finaldata_pivot[['IPP'] + tests + ['Vitamine_D']]

# Suppression des lignes avec des valeurs manquantes
finaldata_pivot = finaldata_pivot.dropna()

# Sélection des caractéristiques et de la cible
features = tests  # Utiliser les tests comme caractéristiques
X = finaldata_pivot[features]
y = finaldata_pivot['Vitamine_D']

# Fonction de prétraitement des données
def preprocess_data(X):
    categorical_cols = ['Sexe']
    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(), categorical_cols)],
        remainder='passthrough'
    )
    X_encoded = ct.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    return X_scaled, ct, scaler

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled, ct, scaler = preprocess_data(X_train)
X_test_scaled = scaler.transform(ct.transform(X_test))

# Optimisation des hyperparamètres avec Grid Search
param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1]
}

grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

print("Meilleurs paramètres trouvés:", grid_search.best_params_)

# Entraînement du modèle SVR avec les meilleurs paramètres trouvés
best_svr = SVR(kernel=grid_search.best_params_['kernel'], C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'])
best_svr.fit(X_train_scaled, y_train)

# Prédiction sur les données de test
y_pred_svr = best_svr.predict(X_test_scaled)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Login failed. Check your username and password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/home')
@login_required
def index_redirect():
    return redirect(url_for('home'))


@app.route('/')
@login_required
def home():
    return render_template('home.html')

@app.route('/vitamine_d')
@login_required
def vitamine_d():
    return render_template('vitamine_d.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    data = request.json
    new_data = pd.DataFrame(data, index=[0])
    new_data_encoded = ct.transform(new_data)
    new_data_scaled = scaler.transform(new_data_encoded)
    prediction = best_svr.predict(new_data_scaled)[0]
    result = {'prediction': prediction}
    return jsonify(result)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Créer les tables si elles n'existent pas déjà
    app.run(debug=True)