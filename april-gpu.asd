;;;; april-gpu.asd

(asdf:defsystem #:april-gpu
  :description "Run April on GPU"
  :author "John Thingstad <jpthing@online.no>"
  :license  "Apache Licence 2.0"
  :version "0.0.1"
  :serial t
  :depends-on (#:vk #:access #:iterate #:cl-ppcre #:yason #:april)
  :components ((:file "package")
	       (:file "april-utils")
               (:file "april-gpu")))
