source "https://rubygems.org"

# Use GitHub Pages gem which includes Jekyll and all supported plugins
gem "github-pages", group: :jekyll_plugins

# Theme
gem "minima"

# Windows and JRuby does not include zoneinfo files
install_if -> { RUBY_PLATFORM =~ %r!mingw|mswin|java! } do
  gem "tzinfo"
  gem "tzinfo-data"
end

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.1.1", :install_if => Gem.win_platform?

# Required for Jekyll 3.0+
gem "webrick", "~> 1.7"
