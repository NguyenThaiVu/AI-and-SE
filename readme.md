# Java Method Dataset Builder

This project crawls Java methods from top GitHub repositories using the GitHub API, then extracts, cleans, and stores them in a structured dataset (*.csv) for software engineering and machine learning research.

---

## Features

* **Repository**: top Java repositories by stars (with license filtering).
* **File & commit tracking**: Collects `.java` file paths and their last commit SHA.
* **Method extraction**: Parses Java files using [`javalang`](https://github.com/c2nes/javalang) to extract:

  * method name
  * start & end lines
  * method signature
  * original code
  * simple code tokens
* **Data cleaning**: Removes borderline cases, including:

  * methods shorter than 3 lines or longer than 100 lines
  * empty or comment-only methods
  * parsing errors or invalid Java files
* **Deduplication**: Removes duplicate methods across repos/files.
* **CSV output**: Saves results as `java_functions_dataset.csv`.

---

## 🔹 Requirements

* Python 3.8+
* GitHub Personal Access Token (set in `.env` as `GITHUB_TOKEN`)
* Dependencies:

  ```bash
  pip install -r requirements
  ```

---

## 🔹 Usage

1. Set your GitHub token in `.env`:

   ```
   GITHUB_TOKEN=your_personal_access_token
   ```
2. Run the script:

   ```bash
   python crawl_java_methods.py
   ```
3. Output dataset will be saved as:

   ```
   java_functions_dataset.csv
   ```

---

## 🔹 Output Schema

The CSV contains:

| Column         | Description                                     |
| -------------- | ----------------------------------------------- |
| repo\_name     | Full name of the repository (`owner/name`)      |
| repo\_url      | GitHub repo URL                                 |
| commit\_sha    | Last commit SHA for the file                    |
| file\_path     | Path to the Java file inside the repo           |
| method\_name   | Extracted method name                           |
| start\_line    | Line where the method starts                    |
| end\_line      | Line where the method ends                      |
| signature      | Method signature (modifiers, return type, args) |
| original\_code | Full code of the method                         |
| code\_tokens   | Simple whitespace-split tokens                  |

---

## 🔹 License Filtering

By default, only repositories with the following licenses are included:

* MIT
* Apache-2.0
* BSD-2-Clause
* BSD-3-Clause
