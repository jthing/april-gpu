;;;; package.lisp

(defpackage #:april-utils
  (:use #:cl #:vk)
  (:export
   #:copy-to-device
   #:with-mapped-memory))

(defpackage #:april-gpu
  (:use #:cl #:vk #:iterate #:access #:april-utils))
