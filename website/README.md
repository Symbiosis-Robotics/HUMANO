# HUMANO project website

Use Node.js 22.12 or newer. From this directory run `npm ci` and `npm run dev` to preview the homepage and Task Explorer. Run `npm run build` to generate the static site in `dist/` and `npm run preview` to inspect the built site.

Edit the two HTML entry points, presentation content in `src/content.json`, styles and interactions in `src/`. Update task metadata, paired demonstrations and evaluation scores in the repository’s `resources/data/`. The scores currently shown are provisional manuscript results pending final v1.0 evaluation.

The directories in `public/resources/` link to the shared branding, images, videos and OOD examples at the repository root. Keep the relative links when cloning. The build packages these files into the deployed site. Record media attribution in `public/THIRD_PARTY_NOTICES.md`.

## GitHub Pages

For the public `Symbiosis-Robotics/HUMANO` repository, select **Settings → Pages → Source → GitHub Actions**. Changes to the website or its shared resources on `main` trigger the Pages workflow. It can also be run manually. The workflow builds the same site in the development repository and limits publication to the public repository.
