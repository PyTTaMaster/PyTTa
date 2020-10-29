# Guidelines 

If this pull request is accepted `.github/workflows/README.md` file can be deleted or edited accordingly.

## Reasoning

1. Currently the last version of PyTTa compiled to pip doesn't correspond
to the best version on the GitHub repository. In order to keep PyPI
aligned to GitHub repo and always updated the `build-publish-to-pypy.yml` GitHub action is suggested.  

    Since the intended goal of this pull request is to maintain PyPI always updated, `setup.py` was changed accordingly,
introducing some classifiers and using the main `README.md` file as the project description for PyPI.

    The main `README.md` file was changed to show the relevant changes. All of them refers to `pip` .
  
2. When installing PyTTa with `pip install pytta` its dependencies aren't installed automatically
because of the old `setup.py` from PyTTa-0.1.0b1, making the user installing them manually.
Since the build workflow uses always the latest `setup.py`,
when an user install PyTTa with `pip install pytta` every listed dependency should now be installed together with PyTTa.

3. To maintain PyTTa PyPI repository as close as possible to the current development branch source code.

## How to setup this workflow

The repository maintainers MUST add a GitHub secret for PyPI at 
 [https://github.com/PyTTAmaster/PyTTa/settings/secrets](https://github.com/PyTTAmaster/PyTTa/settings/secrets).

According to `.github/workflows/build-publish-to-pypi.yml` file:

* PyPI secret name MUST be:

    `Name: pypi_password`
    
    And its value MUST be:
    
    `Value: API TOKEN for PyTTa from https://pypi.org/`
    
GitHub Secrets are stored encrypted.

## How it works 

This workflow affects **only** tagged commits with a tag that starts with letter `v`.

If the pushed commit **is** tagged with a tag **starting** with `v`,
the action triggered will try to build PyTTa distribution package using latest Python stable version.
If the package builds successfully it will be automatically uploaded to PyPI.

This encourages a continuous delivery workflow and allow users to always have the latest release by using 
`$ pip install pytta`.

### Tagged commits

To keep things organized:

`settings{'version': x.y.zbw}`

from setup.py MUST have the same version name as the git tag name,
so that releases are properly tagged with the correct
release version.

When using tagged commits annotated tags are recommended and encouraged but not mandatory.

## Examples

### For annotated tags:

The following syntax is used:

`$ git tag -a <tag-name> -m <tagging-message>`

e.g.: for setup.py containing `settings{'version': '0.1.0b9'}`, the following syntax would be used:

`$ git tag -a v0.1.0b9 -m "simple tagging message"`

By default, the `git push` command doesn’t transfer tags to remote servers,
so you will have to **explicitly** push tags to origin after you have created them. The following syntax is used:

`$ git push origin <tag-name>`

The complete workflow in this example would be:

```
...
$ git commit -m "commit message"
$ git tag -a v0.1.0b9 -m "simple tagging message"
$ git push origin v0.1.0b9

# Alternatively you can use
# $ git push origin <branch> v0.1.0b9
# if you want to push the tagged commit to other branch than the default branch
```

After pushing the tagged commit `build-publish-to-pypi.yml` will be triggered.

### For lightweight tags:

The following syntax is used:

`$ git tag <tag-name>`

The complete workflow would be the same as above but using the lightweight tag syntax
in place of the annotated tag syntax.