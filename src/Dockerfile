FROM ucbjey/risecamp2018-base

# use apt-get as user "root" to install ubuntu packages
USER root

# install graphviz and pydot
RUN apt-get update
RUN apt-get install -y python-pydot python-pydot-ng graphviz

USER $NB_USER

# download and set up Flor
RUN pip3 install graphviz gitpython tqdm
RUN touch ._15.txt
RUN git clone https://github.com/ucbrise/flor
WORKDIR flor
RUN pip3 install -r requirements.txt
RUN pip3 install nltk
RUN pip3 install stop_words
RUN pip3 install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable spellchecker/main
RUN python3 -c 'import nltk; nltk.download("punkt")'
RUN python3 -c 'import nltk; nltk.download("wordnet")'

# download and install grit
WORKDIR /home/$NB_USER
RUN git clone https://github.com/ground-context/grit
RUN git clone https://github.com/ground-context/client
ENV PYTHONPATH $PYTHONPATH:/home/$NB_USER/grit/python:/home/$NB_USER/client/python:/home/$NB_USER/flor

# perform boot-time initialization by adding a startup script
COPY init-world.sh /usr/local/bin/start-notebook.d


USER root
# copy the tutorial into the container.
# do this last so that your container builds are as fast as possible
# when updating the content in tutorial/
COPY flor_cache.d /home/$NB_USER/flor.d/
# COPY grit_cache.d /home/$NB_USER/grit.d/
COPY tutorial /home/$NB_USER/tutorial/
COPY risecamp_demo_out_cache.d /home/$NB_USER/tutorial/risecamp_demo_out.d/
RUN chown -R $NB_USER /home/$NB_USER
