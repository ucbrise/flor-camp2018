#!/usr/bin/env bash

rm -Rf /tmp/flor-camp2018
git clone git@github.com:ucbrise/flor-camp2018.git /tmp/flor-camp2018
rm -Rf /home/$NB_USER/tutorial/
mkdir /home/$NB_USER/tutorial
cp -a ./flor-camp2018/src/tutorial/. ./tutorial/
