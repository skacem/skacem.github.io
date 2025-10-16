source "https://rubygems.org"
# Hello! This is where you manage which Jekyll version is used to run.
# When you want to use a different version, change it below, save the
# file and run `bundle install`. Run Jekyll with `bundle exec`, like so:
#
#     bundle exec jekyll serve
#
# This will help ensure the proper Jekyll version is running.
# Happy Jekylling!
# gem "jekyll"
# This is the default theme for new Jekyll sites. You may change this to anything you like.
gem "minima"
# If you want to use GitHub Pages, remove the "gem "jekyll"" above and
# uncomment the line below. To upgrade, run `bundle update github-pages`.
gem "github-pages", group: :jekyll_plugins
# If you have any plugins, put them here!
# Note: GitHub Pages only supports certain plugins
# Unsupported plugins are commented out
group :jekyll_plugins do
	# gem "jekyll-scholar"  # Not supported by GitHub Pages
	gem "jekyll-sitemap"
	gem "jekyll-feed"
	# gem "jekyll-katex"  # Not supported by GitHub Pages
    # gem "kramdown-syntax-coderay"  # Not supported by GitHub Pages
    # gem "kramdown-math-katex"  # Not supported by GitHub Pages
	gem "jekyll-paginate"
	gem "jemoji"
	# gem 'jekyll-paginate-v2', github: 'sverrirs/jekyll-paginate-v2'  # Not supported by GitHub Pages

end

# Windows and JRuby does not include zoneinfo files, so bundle the tzinfo-data gem
# and associated library.
install_if -> { RUBY_PLATFORM =~ %r!mingw|mswin|java! } do
	gem "tzinfo"
	gem "tzinfo-data"
end

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.1.1", :install_if => Gem.win_platform?

gem "webrick", "~> 1.7"
