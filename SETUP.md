# GitHub Actions Secrets Setup for AQI Predictor

To securely use your AQICN API key in GitHub Actions, you must add it as a repository secret. Follow these steps:

## 1. Go to Your GitHub Repository
- Open your project repository on GitHub (e.g., `https://github.com/your-username/aqi-predictor-lahore`).

## 2. Open Settings
- Click on the **Settings** tab at the top of the repository page.

## 3. Go to Secrets and Variables
- In the left sidebar, click **Secrets and variables** > **Actions**.

## 4. Add a New Repository Secret
- Click the **New repository secret** button.
- For **Name**, enter: `AQICN_API_KEY`
- For **Secret**, paste your AQICN API key (e.g., `81639ed2195e846449b2be120efc8292828c5f59`).
- Click **Add secret** to save.

## 5. Done!
- Your secret is now available to GitHub Actions workflows as `${{ secrets.AQICN_API_KEY }}`.
- No need to commit your API key to the repository.

---

**Note:** Never share your API key publicly or commit it to your codebase. Always use GitHub Secrets for sensitive information.
