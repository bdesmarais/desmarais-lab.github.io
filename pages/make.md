title: GNU Make Tutorial
type: resource

### What is GNU Make?

[GNU Make](http://www.gnu.org/software/make/) is a tool that allows you to construct a [direct acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph) of dependencies amongst the files in your data analysis project (check out the the [Wikipedia](https://en.wikipedia.org/wiki/Make_(software)) article). Make was developed to handle the building of complex pieces of software composed of many source files that need to be compiled or run in a particular order. Often data analysis projects are quite complicated. You might, for example, scrape some data from the web, clean it, do some analysis on it, produce some plots, and then create a presentation and/or paper. Make allows you to provide a "recipe" that describes in what order files need to executed in order to produce a result (a cleaned data file, some figures, a log file, etc.). Each of these build procedures is executed in a seperate instance of your default shell. Makefiles are executed by simply calling `make` from the command line, or, less commonly, by referring to a specific makefile.

Make is language agnostic. This is advantageous because, despite what some language zealots seem to believe, there is not usually one best tool for an entire project. Data analysis often involves shell scripting, a bit of [R](http://www.r-project.org/), [Python](http://www.python.org/), and who knows what else. [STATA](http://www.stata.com/) is still quite dominant in many of the social sciences. Make can handle anything that can be executed from the command line (actually anything, if you use [automator](http://support.apple.com/kb/HT2488) or something like it).

### Reasons to Use Make

Make can encourage you to modularize your code. Rather than having one file `analysis.R` that cleans your data, estimates your model(s), and produces your plots, you can have them split up by task. This makes it easier for others to understand your code and for you to debug it. When you are using R in [batch-mode](http://stat.ethz.ch/R-manual/R-devel/library/utils/html/BATCH.html) `.Rout` files are automatically produced, which is the output from the interpreter, making it easier to see what, if anything, went wrong. Modularizing your code in this manner usually necessitates writing intermediate results to file. For complex objects this can sometimes be a pain (though R makes it easy with `.RData` files, which are automatically produced when R is run in batch-mode).

I think [literate programming](https://en.wikipedia.org/wiki/Literate_programming)  ([sweave](http://www.stat.uni-muenchen.de/~leisch/Sweave/), [knitr](http://yihui.name/knitr/), and [org-babel](http://orgmode.org/worg/org-contrib/babel/) are a few implementations) is great, but I am not convinced it actually makes it easier to reproduce results (most implementations are language specific, though org-babel is not). Using any of these tools makes building a document necessary to see the results of the data anlysis (or any of the intermediate products of the data analysis). The things you need to have installed to produce the document (some flavor of TeX and/or [org-mode](http://orgmode.org/) and [Emacs](http://www.gnu.org/software/emacs/)) are not installed by default on most systems with a Unix command line interface. Make is. With Make you can seperate your data analysis from the document which presents your analysis while retaining the automated integration between the two.

### How to Use Make

The Make model is very easy to understand. There are things to be built (a target) and a list of commands that need to be executed to build said target. Make determines whether a command needs to be executed by checking if the time-modified stamp on the build target is greater than the time-modified stamp of any of the build dependencies. If that is the case, then the target is out of date and needs to be rebuilt. If not, then the build target does not need to be rebuilt, and the commands that need to be run to build the target are not executed. Suppose you've written a Python script that fetches some data, and writes a file, `raw.csv` to disk. You would write:

	:::bash
	raw.csv: get_data.py
		python get_data.py

If `raw.csv` doesn't exist, or was modified at a later time than `get_data.py` (meaning that `get_data.py` has been modified since `raw.csv` was last produced) the command `python get_data.py` is run, which (re)generates `raw.csv`. More realistically you have a number of scripts working together. You might have a shell script that cleans the data, and some R scripts that run some analysis and produce a few plots. Note that the the commands listed below the build target must be indented using the tab character (though you can modify this).

	:::bash
	raw.csv: get_data.py
		python get_data.py

	clean.csv: clean.sh raw.csv
		source clean.sh

	model.Rout: model.R clean.csv
		R CMD BATCH model.R

	plot.Rout: plot.R model.Rout
		R CMD BATCH plot.R

Each build target is specified as reliant on the file that needs to be run to build it in addition to the files necessary to build its dependencies. It is only necessary to specify a first order dependence in this case. Say you've built the entire project, but then you go and change something in `get_data.py`. The next time you build the project this will trigger a rebuild of the Python data retrieving script, which will trigger a re-run of the cleaning shell script, and so on, resulting in a full rebuild. If, howver, you'd made a change in `clean.sh` then you would have triggered a re-run of the model and plotting scripts, but not the cleaning shell script.

Variables (also called macros) are defined using the `:=` operator and accessed (expanded) using `$(var)`. There are actually other ways to define variables in Make (this is a "simply expanded variable"). See [section 6.2](http://www.gnu.org/software/make/manual/make.html#Flavors) in the Make manual to read more about the different "flavors" of variables. Variables are [lazily evaluated](https://en.wikipedia.org/wiki/Lazy_evaluation), meaning that they are only expanded when they are actually called. When compiling a TeX document I will define a variable for the command, and then call it on the relevant dependencies.

	:::bash
	TEXCMD := pdflatex -interaction=batchmode
	paper.pdf: paper.tex
		$(TEXCMD) paper.tex

There are a number of special variables that Make uses. `$<` expands to the first dependency. This is useful when you have to call commands that involve the same file several times, as you typically have to do with compiiling a TeX document. Assuming you've already defined `TEXCMD` as above:

	:::bash
	paper.pdf: paper.tex
		$(TEXCMD) $<
		$(TEXCMD) $<
		bibtex *.aux
		$(TEXCMD) $<

This runs TeX twice, then bibtex (on the only `.aux` file that should be there), and then runs TeX another time for good measure. You can of course add your figures (or, better yet, the log file that is produced by your plotting script) as a dependency for your paper. Other special variables include `$@` which expands the build target, and `$?` which expands to any dependencies which have a time stamp more recent than the target.

You can define dummy targets in addition to the variety already described. A dummy target is a target with no commands directly associated with it (it is sort of a meta-target). For example if you have a couple of scripts that involve data acquisition and cleaning, another few that involve data analysis, and a few that involve the presentation of results (paper, plot), then you might define a dummy for each of them. You can be even more meta if you want, and define a dummy target that depends on dummy targets. For example:

	:::bash
	all: data model paper
	data: raw.csv
	model: model.Rout
	paper: plot.Rout paper.pdf

	raw.csv: get_data.py
		python get_data.py

	clean.csv: clean.sh raw.csv
		source clean.sh

	model.Rout: model.R clean.csv
		R CMD BATCH model.R

	plot.Rout: plot.R model.Rout
		R CMD BATCH plot.R

	paper.pdf: paper.tex
		$(TEXCMD) $<
		$(TEXCMD) $<
		bibtex *.aux
		$(TEXCMD) $<

Assuming that you've named the file "makefile" or "Makefile" or something like that, simply typing "make" at the command line while inside your project's directory will execute the build process. You can of course refer to a specific target, e.g. `make paper.pdf`. This is especially useful with the dummy targets that just discussed, as it is often the case that you'll want to rebuild only part of the whole.

With  Make it will be both easier to manage your own work (especially if you "build" your projects on remote servers frequently) and easier for other people to reproduce it. To reiterate a bit, it will be easier to reproduce because your code will (hopefully) be more modular and easier to understand, because all the dependencies amongst your files will be explicitly stated, and you won't be tying the results from your analysis to the compiling (or whatever) of your presentation or paper. Since Make is distributed with just about every operating system that has a Unix shell (there is a version of Make for Windows), the dependency is usually fulfilled without any effort on the part of the user.

[Rob Hyndman](http://robjhyndman.com/hyndsight/makefiles/) (a statistician interested in forecasting) also uses Make in much the same way. [Mike Bostock](http://bost.ocks.org/mike/make/) also has a cool post about using Make. So does [Matt Might](http://matt.might.net/articles/intro-to-make/), who also points out that [O'Reilly](http://oreilly.com/catalog/make3/book/index.csp) has an open-book on Make as well. The [manual](http://www.gnu.org/software/make/manual/make.html) or manpages are the real authoritative source of information on the matter of course.
