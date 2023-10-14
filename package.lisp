;;;; package.lisp

(defpackage #:april-utils
  (:use #:cl #:vk)
  (:export
   #:copy-to-device
   #:with-mapped-memory
   #:buffer-size))

(defpackage #:april-gpu
  (:use #:cl #:vk #:iterate #:access #:april-utils)
  (:local-nicknames (:au :april-utils)))
