# Java Methods Dataset Builder

## Step 1 – Crawl Java Repos

Run the first section of the notebook to **download and unzip** open-source Java repositories into `java_repos/`.

## Step 2 – Parse with javalang

Run the second section to **parse all `.java` files** and save methods into `methods.csv` with details like:
`repo_name, file_path, method_name, start_line, end_line, signature, original_code`