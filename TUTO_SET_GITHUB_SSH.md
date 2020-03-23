#TUTO BEAUTÉ GITHUB

# setting git
1. install git on your computer
2. open terminal
3. set your configuration
```
git config --global user.name "John Doe"
git config --global user.email johndoe@example.com
```

# create repo (if you want a NEW repo)
1. create in GitHub
2. `git init`
3. `git add`
4. `git commit -m "my message"`
5. `git remote add origin git@github.com:username/new_repo`
6. `git push --set-upstream origin master` (wont work until ssh done)
7. read set ssh if it doesnt work and try 6. again

# set ssh  (if it is the first time you access GitHub or have a access denial)
0. read https://help.github.com/en/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
1. open terminal/git bash
2. `create key $ ssh-keygen -t rsa -b 4096 -C "your_email@example.com"`
3. go to your users/.ssh/ folder (hidden folder)
4. open and copy content of users/.ssh/id_rsa.pub
5. go to https://github.com/settings/keys
6. new SSH key
7. give a title for your machine and paste the id_rsa.pub content

# Get the repo on your machine and start to work

1. In terminal or Git Hash
  `cd your_local_git_folder`
  `git clone git@github.com:zetadiagnostic/TinnitusEEG.git`
2. Go into the new folder 
	`cd TinnitusEEG`
	`git status` (it should return that you are up to date)

# Git Branch locally and put it on github GitHub (if you want develop new features)

0. (optional) To make things easier, and have the same name locally and remote without setting the upstream
 `git config --global push.default current`
1. Create a new branch:
    `git checkout -b feature_branch_name`
2. Edit, add and commit your files.
3. Push your branch to the remote repository:
    (normal) `git push -u origin feature_branch_name`
    (if you done step 0.) `git push -u`

# Get a existing branch on GitHub that is NOT yet on your machine locally
0. to check all branch (even the remote):
    `git branch -a` (you should see the branch that you want e.g. "origin/feature_branch_remotename")
1.  Get the remote branch locally
	`git checkout -b feature_branch_localname origin/feature_branch_remotename`
	example: `git checkout -b test origin/test`

# Git merge branch into master (or other)

1. (better than the option below). USE PULL REQUEST FROM GITHUB AND ASK A CODE REVIEW (assign someone)

OR (only for small changes that doesn't require a pull request and code review)

1. `git checkout master`(this is the output folder)
2. `git merge feature_branch_name`(will merge the feature_branch into master)
3. If you don't need the feature_branch anymore, you can delete it:
   `git branch -d feature_branch_name`
