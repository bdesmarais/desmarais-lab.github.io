MSG='automatic update'

build:
	python3 app.py build
	rsync -r ./build/ ./

test:
	python3 app.py

publish:
	python3 app.py build
	rsync -r ./build/ ./
	rm -rf build
	git add *
	git commit -m $(MSG)
	git push origin master



