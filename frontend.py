'''
See
https://stackoverflow.com/questions/45553629/how-to-run-a-python-script-on-flask-backend
'''

from flask import Flask, render_template, request
from textstyletransfer import transfer

app = Flask(__name__)


@app.route('/')
def display():
    return render_template('index.html', f_sent='')


@app.route('/styleTransfer', methods=['GET', 'POST'])
def styleTransfer():
    if request.method == 'POST':
        c_sent = request.form['c_sent']
        s_sent = request.form['s_sent']
        error = None

        if not c_sent:
            error = 'Content sentence is required.'
        elif not s_sent:
            error = 'Style sentence is required.'

        if error is None:
            f_sent = transfer(c_sent, s_sent)

        return render_template('index.html', f_sent=f_sent)


if __name__ == '__main__':
    app.run(debug=True)
