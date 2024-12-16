# Beofre commit
1. `cd <path-of-pyq> && black -l 120 ./pyq/` 
2. `isort ./pyq/ --line-length 120`
3. `flake8 --max-line-length=120 ./pyq/`
4. `python -m unittest test`

# Contribute to the repository
1. Add an issue to the repository with the name of the bug/feature that you're going to fix/add
2. Make a branch with the name `fix_bug/<issue-id>-<your_bug_name>` or `feature/<issue-id>-<your_feature_name>`
3. Push your code to the brach that you created
4. Submit a merge request to the master branch
