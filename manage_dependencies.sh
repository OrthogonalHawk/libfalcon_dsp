#!/bin/bash

NUM_OPTS_SET=0
DOWNLOAD_DEPS=""
LIST_DEPS=""

USAGE="Usage: $0 [command]
  Only one of the following commands/options may be used at once.
  -d, --download      Download dependencies; downloads repos to directories in ../
  -h, --help          Display help information
  -l, --list          List dependencies found in the README.md file"
  
TEMP=`getopt -o dhl --long download,help,list -n $0 -- "$@"`
if [ $? != 0 ]; then
    exit 1
fi
eval set -- "$TEMP"
while true; do
    case "$1" in
        -d|--download)
          DOWNLOAD_DEPS="--download"
          let "NUM_OPTS_SET++"
          shift
          ;;
    
        -l|--list)
          LIST_DEPS="--force"
          let "NUM_OPTS_SET++"
          shift
          ;;
        
        -h|--help)
          echo "$USAGE"
          shift
          exit 0
          ;;
          
               --)
          shift
          break
          ;;
                *)
          echo "Internal Error!"
          exit 1
          ;;
    esac
done

# make sure that the user provided one or more command options
if [ $NUM_OPTS_SET = 0 ] || [ $NUM_OPTS_SET -gt 1 ]; then
    echo "$USAGE"
    exit 0
fi

CUR_WORKING_DIR=$(pwd)

# parse the README.md file to download any required dependencies
APP_OR_LIBRARY_README_MD_FILE=README.md
BUILD_DEPENDENCY_MAGIC_STR="REQUIRED_BUILD_DEPENDENCY"

GIT_REPO_NAME_MAGIC_STR="replace_with_git_repo_name"
GIT_CLONE_STR="git@github.com:OrthogonalHawk/$GIT_REPO_NAME_MAGIC_STR.git"

# verify that the required file exists
if [ ! -f $APP_OR_LIBRARY_README_MD_FILE ]; then
    SCRIPT_NAME=$(basename "$0")
    
    printf "ERROR: unable to find $APP_OR_LIBRARY_README_MD_FILE; the $SCRIPT_NAME script must be run from the repo root directory\n"
    exit 1
fi

# go through each one of the dependencies and download the required Git repo
grep $BUILD_DEPENDENCY_MAGIC_STR $APP_OR_LIBRARY_README_MD_FILE | while read -r DEPENDENCY_LINE ; do

    REPO_NAME_AND_LINK="$(echo $DEPENDENCY_LINE | cut -d'|' -f2)"
    REPO_NAME="$(echo $REPO_NAME_AND_LINK | cut -d'[' -f2 | cut -d']' -f1)"
    REPO_LINK="$(echo $DEPENDENCY_LINE | cut -d'(' -f2 | cut -d')' -f1)"
    REPO_TAG="$(echo $DEPENDENCY_LINE | cut -d'|' -f3)"
    
    REPO_LINK_FOR_CLONE="$(echo $GIT_CLONE_STR | sed "s=${GIT_REPO_NAME_MAGIC_STR}=${REPO_NAME}=g")"
    REPO_CHECKOUT_LOCATION=../$REPO_NAME
    
    if [ ! -z $LIST_DEPS ]; then
        printf "repo: %-48s    tag:%-48s \n" "$REPO_NAME" "$REPO_TAG"
    fi
    
    if [ ! -z $DOWNLOAD_DEPS ]; then
    
        if [ -d $REPO_CHECKOUT_LOCATION ]; then
        
            # directory already exists; make the assumption that it is a Git repo
            #  and can be updated with the latest contents from the remote server
            printf "Updating dependency at $REPO_CHECKOUT_LOCATION\n"
            cd $REPO_CHECKOUT_LOCATION
            
            if ! git fetch; then
                printf "ERROR: unable to do 'git fetch' from $REPO_CHECKOUT_LOCATION; please cleanup directory and retry\n"
                exit 1
            fi
            
            if ! git checkout $REPO_TAG; then
                printf "ERROR: unable to checkout tag $REPO_TAG from $REPO_NAME\n"
                exit 1
            fi
            
        else
            
            # since the directory does not exist it will be created during the Git
            #  clone operation
            if ! git clone $REPO_LINK_FOR_CLONE $REPO_CHECKOUT_LOCATION; then
                printf "ERROR: do old dependencies exist? cleanup and retry\n"
                exit 1
            fi
            
            cd $REPO_CHECKOUT_LOCATION
            
            if ! git checkout $REPO_TAG; then
                printf "ERROR: unable to checkout tag $REPO_TAG from $REPO_NAME\n"
                exit 1
            fi
            
        fi
        
        cd $CUR_WORKING_DIR
    fi
    
done