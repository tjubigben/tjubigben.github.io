# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a personal blog built with Jekyll using the [Not Pure Poole](https://github.com/vszhub/not-pure-poole) theme, hosted on GitHub Pages at `https://tjubigben.github.io`. Blog posts are primarily written in Chinese.

## Common Commands

```sh
# Install dependencies
bundle install

# Start local development server (includes drafts)
./scripts/serve
# equivalent: bundle exec jekyll serve --draft --trace

# Create a new draft post
./scripts/draft "Post Title"
# equivalent: bundle exec jekyll draft "Post Title"

# Publish a draft
./scripts/publish _drafts/post-filename.md
# equivalent: bundle exec jekyll publish _drafts/post-filename.md

# Build the site
bundle exec jekyll build
```

## Blog Post Format

Posts live in `_posts/` with filenames like `YYYY-MM-DD-slug.md`. Drafts live in `_drafts/`.

Standard frontmatter:

```yaml
---
layout: post
title: "Post Title"
date: YYYY-MM-DD
categories: [Category1, Category2]
tags: [tag1, tag2]
author: 王凯
excerpt: "Short description shown in post listings."
toc: true      # optional: show table of contents on right side
math: true     # optional: enable MathJax for math rendering
---
```

## Site Structure

- `_posts/` — published blog posts (Markdown)
- `_drafts/` — unpublished draft posts
- `_layouts/` — page layout templates (HTML)
- `_includes/` — reusable HTML partials
- `_sass/` — SCSS stylesheets
- `_data/` — site data files:
  - `navigation.yml` — top navigation links
  - `social.yml` — social media links (Font Awesome icons)
  - `archive.yml` — archive page configuration (dates/categories/tags)
- `assets/styles.scss` — main stylesheet entry point
- `_config.yml` — Jekyll configuration

## Theme Customization

The theme is loaded via `jekyll-remote-theme: vszhub/not-pure-poole`. To customize:

- **Custom head content**: create `_includes/custom-head.html`
- **Color themes**: modify CSS variables in `_sass/_variables.scss` using `[data-theme="name"]` selectors
- **Cover image**: set `cover_image` in `_config.yml` or per-page frontmatter
- **Navigation**: edit `_data/navigation.yml`
- **Social links**: edit `_data/social.yml` (icon values are Font Awesome classes)

## Deployment

The site deploys automatically to GitHub Pages when changes are pushed to the `main` branch. The `CNAME` file sets the custom domain.
