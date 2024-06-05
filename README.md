# Personal site
Im using [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) which is a flexible, easy to implement [Jekyll](https://jekyllrb.com/) theme.

Installing ruby with mac could be a pain in the ass.
You can clone the Minimal Mistakes default, and then run the following for running locally:

`docker run -p 4000:4000 -v $(pwd):/site bretfisher/jekyll-serve`

build locally:
`docker run -p 4000:4000 -v $(pwd):/site bretfisher/jekyll-serve jekyll build`
