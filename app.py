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
app.config['FREEZER_DESTINATION'] = '/'
app.config['FREEZER_BASE_URL'] = 'https://desmarais-lab.github.io/'
app.config['FREEZER_RELATIVE_URLS'] = True
FLATPAGES_MARKDOWN_EXTENSIONS = ['codehilite', 'footnotes', 'fenced_code']

pages = [p for p in flatpages if p.meta.get('type') == 'resource']

@app.route('/')
def index():
    return render_template('index.html', page=flatpages.get('home'))

@app.route('/people/')
def people():
    return render_template('people.html')

@app.route('/projects/')
def projects():
    return render_template('projects.html')

@app.route('/publications/')
def publications():
    return render_template('publications.html')

@app.route('/resources/')
def resources():
    return render_template('resources.html', pages=pages)

@app.route('/<path:path>/')
def page(path):
    page = flatpages.get_or_404(path)
    return render_template('page.html', page=page)

# @freezer.register_generator
# def page_generator():
#     yield '/make/'

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        freezer.freeze()
    else:
        app.run(port=8000)
