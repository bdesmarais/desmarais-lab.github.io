import sys
from flask import Flask, render_template
from flask_flatpages import FlatPages
from flask_frozen import Freezer

DEBUG = True
FLATPAGES_AUTO_RELOAD = DEBUG
FLATPAGES_EXTENSION = '.md'

app = Flask(__name__)
app.config.from_object(__name__)
flatpages = FlatPages(app)
freezer = Freezer(app)
app.config['FREEZER_BASE_URL'] = ''


@app.route('/')
def index():
    return render_template('index.html', page=flatpages.get('home'))

@app.route('/<path:path>/')
def page(path):
    page = flatpages.get_or_404(path)
    return render_template('page.html', page=page)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        freezer.freeze()
    else:
        app.run(port=8000)
