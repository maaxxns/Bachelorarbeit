all: build/dissfin.pdf


TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build
                                                                                
build/dissfin.pdf: FORCE | build
	latexmk $(TeXOptions) DISS.TEX
	
FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
