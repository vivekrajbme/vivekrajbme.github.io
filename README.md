# vivekrajbme.github.io: portfolio website

Live site: https://vivekrajbme.github.io. It is built by GitHub Pages from this repository with
the Jekyll "Minimal Mistakes" remote theme. It costs nothing to host.

## What's here

| Path | What it holds |
|---|---|
| `_config.yml` | Site title, description, author sidebar (name, bio, links), theme settings |
| `_data/authors.yml` | Author profile and social links: LinkedIn vivekraj-cvml, GitHub vivekrajbme, Scholar, ORCID |
| `_data/navigation.yml` | Top menu links |
| `_pages/` | Standalone pages, e.g. `contact.md` (email dlforvivek@gmail.com, LinkedIn, ORCID) |
| `_posts/` | Blog/project posts, named `YYYY-MM-DD-title.md` |
| `index.md` | Home page |
| `assets/` | Images and files (put your resume PDF or figures here) |
| `Gemfile` | Only needed to preview the site locally |

Recent changes: LinkedIn updated to vivekraj-cvml, ORCID added, and contact email set to
dlforvivek@gmail.com (29 Sep 2026).

## How to update the site (step by step)

1. Edit a Markdown file. For example, to add a project post, create
   `_posts/2026-10-05-real-time-limb-segmentation.md` starting with:
   ```
   ---
   title: "Real-time residual-limb segmentation"
   excerpt: "MobileNetV2-FPN distilled from SAM 2.1, under 90 ms closed loop"
   ---
   Your text here. Images: ![caption](/assets/images/fig1.png)
   ```
2. Publish:
   ```powershell
   cd F:\JobSearch\vivekrajbme.github.io
   git add .
   git commit -m "Add project post: limb segmentation"
   git push
   ```
3. Wait 1–2 minutes, then refresh the live site. If it doesn't change, open the repo on GitHub,
   go to **Actions**, and read the error of the "pages build" run. It is usually a YAML mistake
   in the `---` header.

Keep the same honesty rules as the resumes: no robotics experience claimed, and "individuals with
transradial amputation", never "amputee".
