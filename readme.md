# Java Method Dataset Builder

This project crawls Java methods from top GitHub repositories using the GitHub API, then extracts, cleans, and stores them in a structured dataset (*.csv) for software engineering and machine learning research.

---

## 1. Features

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
* **License Filtering**: only repositories with the following licenses are included: MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause
* **CSV output**: Saves results as `java_functions_dataset.csv`.

### Code Tokenization

In this project, we use a **regex-based tokenizer** to process Java source code into meaningful tokens.
Unlike plain space-based splitting, this tokenizer preserves the **syntactic structure of code**, which is crucial for analysis and machine learning tasks.

### How it works

The tokenizer breaks code into:

* **Identifiers** (e.g., `main`, `System`, `args`)
* **Keywords** (e.g., `public`, `class`, `return`)
* **Numbers** (e.g., `42`)
* **Strings** (e.g., `"hello"`)
* **Operators** (e.g., `==`, `!=`, `<=`, `&&`)
* **Punctuation** (e.g., `{`, `}`, `(`, `)`, `;`)

**Example**

```java
public static void main(String[] args) { return 5; }
```

* **Space tokenization** →

  ```
  ['public', 'static', 'void', 'main(String[]', 'args)', '{', 'return', '5;', '}']
  ```
* **Regex code tokenization** →

  ```
  ['public', 'static', 'void', 'main', '(', 'String', '[', ']', 'args', ')', '{', 'return', '5', ';', '}']
  ```

**Advantages**

* **Preserves syntax**: operators and braces are captured as separate tokens.
* **Cleaner tokens**: avoids mixing symbols with identifiers (e.g., `args)` → `args`, `)`).
* **Model-friendly**: improves consistency for training ML models on code.


---

## 2. Requirements

* Python 3.8+
* GitHub Personal Access Token (set in `.env` as `GITHUB_TOKEN`)
* Dependencies:

  ```bash
  pip install -r requirements
  ```

---

## 3. Usage

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

## 4. Output Data

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

