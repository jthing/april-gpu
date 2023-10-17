;;;; package.lisp

(defpackage #:april-utils
  (:use #:cl)
  (:local-nicknames (#:vk #:vk))
  (:export
   #:with-mapped-memory
   #:with-destructor
   #:copy-to-device
   #:copy-from-device
   #:buffer-size))

(defpackage #:april-gpu
  (:use #:cl)
  (:local-nicknames
   (#:vk #:vk)
   (#:au #:april-utils))
  (:export
   #:make-compute-pipeline-info
   #:make-compute-shader-info
   #:create-compute-pipeline
   #:cleanup-compute-pipeline))
