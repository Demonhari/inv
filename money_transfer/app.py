from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_bootstrap import Bootstrap
from flask_dance.contrib.google import make_google_blueprint, google

app = Flask(__name__, static_folder="templates", static_url_path="/templates")
app.secret_key = 'your_very_secret_key_here'
Bootstrap(app)
blueprint = make_google_blueprint(
    client_id='your_client_id',
    client_secret='your_client_secret',
    scope=["profile", "email"]  # Requested user data
)
app.register_blueprint(blueprint, url_prefix="/login")

# I still need the structure to be flexible 
users = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']  
        question1 = request.form['question1']
        question2 = request.form['question2']
        #re error still coming up, Adding validation and error checking here
        users[username] = {
            'password': password,
            'question1': question1,
            'question2': question2,
            'transfers': []  # can be used to store history, need to think an optimisd way to implement this 
        }
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.get(username)
        if user and user['password'] == password:
            session['username'] = username
            return redirect(url_for('dashboard'))
        return 'Invalid credentials'
    return render_template('login.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        # Process form data and send password reset email
        # ... (code for handling password reset logic)
        return redirect(url_for('login'))
    else:
        return render_template('forgot_password.html')


@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/transfer', methods=['GET', 'POST'])
def transfer():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        amount = request.form['amount']
        transfer_date = request.form['transfer_date']
        users[session['username']]['transfers'].append({
            'amount': amount,
            'date': transfer_date
        })
        return redirect(url_for('transfer_confirmation'))
    return render_template('transfer.html')

@app.route('/transfer_confirmation')
def transfer_confirmation():
    if 'username' not in session:
        return redirect(url_for('login'))
    return 'Transfer submitted successfully!'

@app.route('/transfer_history')
def transfer_history():
    if 'username' not in session:
        return redirect(url_for('login'))
    transfers = users[session['username']].get('transfers', [])
    return render_template('transfer_history.html', transfers=transfers)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/')
def index():
    if not google.authorized:
        return '<a href="/login/google">Login with Google</a>'
    else:
        resp = google.get("/oauth2/v2/userinfo")
        user_info = resp.json()
        return f'Hello, {user_info["name"]}!'
    
@app.route('/login/callback')
def login_callback():
    google.authorized = True
    return redirect('/')  # Redirect to the desired page after login


if __name__ == '__main__':
    app.run(debug=True)
