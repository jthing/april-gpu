;;;; april-gpu.lisp
;;;; Author: John Thingstad
;;;; Creation Date: oct. 2 2023
;;;; Short Description: Setup a compute shader pipeline to allow april to run computations on a GPU.

(in-package #:april-gpu)

;;;-----------------------------------------------------------------------------
;;;
;;; Setup compute pipeline
;;;
;;; Based on: Minimal vulcan compute shader
;;; Author:   İlker Işık aka. necrashter
;;;
;;; Blog entry: https://necrashter.github.io/ceng469/
;;;             project/compute-shaders-intro
;;;
;;; Slides at:  https://docs.google.com/presentation/
;;;             d/1KrDE7a3WT551gjLwa0OO09kgZ71lKJf-6CR6kVVeVAM/
;;;             edit#slide=id.g133da4b698b_0_1059
;;;
;;; Code at:    https://github.com/necrashter/minimal-vulkan-compute-shader
;;;
;;; This code uses the common lisp vulcan interface vk
;;; vk-samples provides example code used to implement the spesifics
;;;
;;;-----------------------------------------------------------------------------

;;--------------------------------------------------
;;  Summary
;;--------------------------------------------------
;;  1. Fill application info
;;  2. Enable validation layer and create instance
;;  3. Physical device
;;  4. Queue family
;;  5. Device
;;  6. Create buffers
;;  7. Query memory types
;;  8. Allocate memory
;;  9. Write data to memory
;; 10. Bind buffers to memory
;; 11. Read shader & Create shader module
;; 12. Descriptor set layout
;; 13. Pipeline layout
;; 14. Pipeline
;; 15. Desciptor pool
;; 16. Descriptor sets
;; 17. Utdate descriptor sets to use our buffers
;; 18. Submit work to GPU
;; 19. Command pool
;; 20. Command buffer
;; 21. Record commands
;; 22. Submit work and wait
;; 23. Read output
;; 24. Cleanup
;;--------------------------------------------------

;;--------------------------------------------------
;; Global constants
;;--------------------------------------------------

(defparameter *debug-pipeline* t)
(defparameter *step* t)

(defvar *application-name* "April GPU")
(defvar *application-version* (list 1 0 0))
(defvar *engine-name* "vk")
(defvar *engine-version* (list 1 0 0))
(defvar *api-version* vk:+header-version-complete+)

(defparameter *enabled-layers* (list "VK_LAYER_KHRONOS_validation"))

;;--------------------------------------------------
;; Exception hirarchy
;;--------------------------------------------------

(define-condition error-compute-pipeline (error) ())

(define-condition error-create-instance (error-compute-pipeline) ())
(define-condition error-physical-device (error-compute-pipeline) ())
(define-condition error-queue-family (error-compute-pipeline) ())
(define-condition error-create-device (error-compute-pipeline) ())
(define-condition error-create-buffers (error-compute-pipeline) ())
(define-condition error-query-memory-types (error-compute-pipeline) ())
(define-condition error-allocate-memory (error-compute-pipeline) ())
(define-condition error-write-data-to-memory (error-compute-pipeline) ())
(define-condition error-bind-buffers-to-memory (error-compute-pipeline) ())
(define-condition error-read-shader (error-compute-pipeline) ())
(define-condition error-create-shader-module (error-compute-pipeline) ())
(define-condition error-descriptor-set-layout (error-compute-pipeline) ())
(define-condition error-pipeline-layout (error-compute-pipeline) ())
(define-condition error-pipeline (error-compute-pipeline) ())
(define-condition error-desciptor-pool (error-compute-pipeline) ())
(define-condition error-descriptor-sets (error-compute-pipeline) ())
(define-condition error-utdate-descriptor-sets-to-use-our-buffers (error-compute-pipeline) ())
(define-condition error-submit-work-to-GPU (error-compute-pipeline) ())
(define-condition error-command-pool (error-compute-pipeline) ())
(define-condition error-command-buffer (error-compute-pipeline) ())
(define-condition error-record-commands (error-compute-pipeline) ())
(define-condition error-submit-work-and-wait (error-compute-pipeline) ())
(define-condition error-read-output (error-compute-pipeline) ())

;;--------------------------------------------------
;; compute-pipeline-info
;;
;; A class created to contain the data used by create-compute-shader
;;
;; You need:
;;  shader object 
;;  input array for the initial data
;;  output array result of applying the compute shader 
;;
;;--------------------------------------------------

(defclass compute-pipeline-info ()
  ((a-shader :initarg :shader
	   :accessor cpi-shader)
   (a-input  :initarg :input
	   :accessor cpi-input)
   (a-output :initarg :output
	   :accessor cpi-output)))

(defun make-compute-pipeline-info (&key shader input output)
  (make-instance 'compute-pipeline-info :shader shader :input input :output output))

;;--------------------------------------------------
;; compute-shader-info
;;--------------------------------------------------

(defclass compute-shader-info ()
  ((instance
    :initform nil
    :accessor csi-instance)
   (device
    :initform nil
    :accessor csi-device)
   (input-buffer
    :initform nil
    :accessor csi-input-buffer)
   (output-buffer
    :initform nil
    :accessor csi-output-buffer)))

(defun make-compute-shader-info ()
  (make-instance 'compute-shader-info))

;;--------------------------------------------------
;; Compute pipeline
;;--------------------------------------------------

;; do-read-output
;; do-submit-work-and-wait
;; do-record-commands
;; do-command-buffer
;; do-command-pool
;; do-submit-work-to-GPU
;; do-utdate-descriptor-sets-to-use-our-buffers
;; do-descriptor-sets
;; do-desciptor-pool
;; do-pipeline
;; do-pipeline-layout
;; do-descriptor-set-layout
;; do-read-shader-&-Create-shader-module
;; do-bind-buffers-to-memory
;; do-write-data-to-memory
;; do-allocate-memory
;; do-query-memory-types

(defun do-create-buffers (shader-info compute-pipeline-info)
  (declare (ignore compute-pipeline-info))
  (let ((my-buffer-create-info
	  (vk:make-buffer-create-info
	   :size 80
	   :usage :vertex-buffer
	   :sharing-mode :exclusive
	   :queue-family-indices (list 1))))
    (let ((input-buffer (vk:create-buffer (csi-device shader-info) my-buffer-create-info))
	  (output-buffer (vk:create-buffer (csi-device shader-info) my-buffer-create-info)))
      (unless (and input-buffer output-buffer)
	(error (make-condition 'error-create-buffers)))
      (when *debug-pipeline*
	(print my-buffer-create-info)
	(when *step* (break)))
      (setf (csi-input-buffer shader-info) input-buffer)
      (setf (csi-output-buffer shader-info) output-buffer))))

(defun do-create-physical-device (shader-info compute-pipeline-info)
  (let* ((my-physical-device (first (vk:enumerate-physical-devices (csi-instance shader-info))))
	 (my-queue-family-properties (vk:get-physical-device-queue-family-properties my-physical-device))
	 (my-compute-queue-family-index
	   (position-if (lambda (q)
			  (member :compute (vk:queue-flags q)))
			my-queue-family-properties))
	 (my-queue-priority 0.0)
	 (my-device-queue-create-info (vk:make-device-queue-create-info
				       :queue-family-index my-compute-queue-family-index
				       :queue-priorities (list my-queue-priority)))
	 (my-device-create-info (vk:make-device-create-info
				 :queue-create-infos (list my-device-queue-create-info)
				 :enabled-layer-names *enabled-layers*)))
    (multiple-value-bind (my-device status)
	(vk:create-device my-physical-device my-device-create-info)
      (when *debug-pipeline*
	(print (vk:get-physical-device-properties my-physical-device))
	(print my-queue-family-properties)
	(print my-device-create-info)
	(when *step* (break)))
      (unless (eql status :success)
	(error (make-condition 'error-create-device)))
      (setf (csi-device shader-info) my-device)
      (do-create-buffers shader-info compute-pipeline-info))))

(defun do-create-instance (shader-info compute-pipeline-info)
  (let* ((my-application-info (vk:make-application-info
			       :application-name *application-name*
			       :application-version (apply #'vk:make-version *application-version*)
			       :engine-name *engine-name*
			       :engine-version (apply #'vk:make-version *engine-version*)
			       :api-version *api-version*))
	 (my-create-info (vk:make-instance-create-info
			  :application-info my-application-info
			  :enabled-layer-names *enabled-layers*)))
    (multiple-value-bind (my-instance status)
	(vk:create-instance my-create-info)
      (when *debug-pipeline*
	(print my-create-info)
	(if *step* (break)))
      (unless (eql status :success)
	(error (make-condition 'error-create-instance)))
      (setf (csi-instance shader-info) my-instance)
      (do-create-physical-device shader-info compute-pipeline-info))))

(defun create-compute-pipeline (compute-pipeline-info)
  "Execute the compute shader in compute-pipeline-info on the GPU.
The input and output arrays are set in compute-pipeline-info.
Returns compute-shader-info (NB! not compute-pipeline-info)."
  (let ((shader-info (make-compute-shader-info)))
    (handler-case (do-create-instance shader-info compute-pipeline-info)
      (error-compute-pipeline (e)
	(format *error-output* "create-compute-pipeline: ~S" (class-name e))
	(cleanup-compute-pipeline shader-info)
	(return-from create-compute-pipeline (values nil :failure))))
    (values shader-info :success)))

(defun cleanup-compute-pipeline (compute-shader-info)
  (let ((info compute-shader-info))
    ;; info could still be alive when this returns. Remeber the repl * ** and ***
    ;; Therefore set the destroyed objects to nil
    (when (csi-device info)
      (vk:destroy-device (csi-device info))
      (setf (csi-device info) nil))
    (when (csi-instance info)
      (vk:destroy-instance (csi-instance info))
      (setf (csi-instance info) nil))))

(defun test-compute-pipeline ()
  (let* ((my-input-vector
	   (make-array 10 :element-type 'fixnum
			  :initial-contents
			  (loop for i from 1 to 10 collecting i)))
	 (my-output-vector
	   (make-array 10 :element-type 'fixnum :initial-element 0))
	 (my-cpu-vector
	   (make-array 10 :element-type 'fixnum
			  :initial-contents
			  (loop for i from 1 to 10 collecting (* i i))))
	 (my-shader-path "./shaders/")
	 (my-shader-file "square.comp.spv")
	 (my-shader-object
	   (vk-utils:read-shader-source
	    (concatenate 'string my-shader-path my-shader-file)))
	 (my-compute-pipeline-info
	   (make-compute-pipeline-info
	    :shader my-shader-object
	    :input my-input-vector
	    :output my-output-vector))
	 (my-compute-pipeline (create-compute-pipeline my-compute-pipeline-info)))
    (format t "Input:         ~{~a~^, ~}~%" (coerce (cpi-input  my-compute-pipeline-info) 'list))
    (format t "Output shader: ~{~a~^, ~}~%" (coerce (cpi-output my-compute-pipeline-info) 'list))
    (format t "Reference CPU: ~{~a~^, ~}~%" (coerce my-cpu-vector 'list))
    (if (equalp (cpi-output my-compute-pipeline-info) my-cpu-vector)
	(format t "Correlation: OK.~%")
	(format t "Correlation: FAIL!~%"))
    (cleanup-compute-pipeline my-compute-pipeline)))

;; (with-instance (my-instance)
;; 	(with-device (my-device)
;; 	  (flet ((my-read-shader-file (shader-file)
;;              (let ((my-file-path (merge-pathnames
;;                                file-name
;;                                (asdf:system-relative-pathname
;;                                 'shaders
;;                                 path)))))
;; 		   (vk-utils:read-shader-source file-path))))
;; 	  (let* ((my-vertex-shader-module-create-info
;; 		   (vk:make-shader-module-create-info
;; 		    :code (my-read-shader-file vertex-shader-file)))
;; 		 (my-fragment-shader-module-create-info
;; 		   (vk:make-shader-module-create-info
;; 		    :code (my-read-shader-file my-fragment-shader-file)))
;; 		 (vertex-shader-module (vk:create-shader-module device vertex-shader-module-create-info)))
;; 	    (let ((my-fragment-shader-module (vk:create-shader-module device my-fragment-shader-module-create-info)))
;; 		  (with-shader-module (my-fragment-shader-module)
;; 		    )
;; 	      ))))

;; find-physical-device-limits (device)
;;   (vk:get-physical-device-properties device))

;; (defun init-compute-pipeline(device layout stage)
;;   (let ((compute-pipeline (vk:make-compute-pipeline-create-info layout stage)))
;;     (vk:create-compute-pipelines device compute-pipeline)))

;; (defun create-compute-pipeline (device set-layouts)
;;   "Vanilla VkPipelineCreateInfo"
;;   (let ((layout-info (vk:make-pipeline-layout-create-info set-layouts))))
;;   (vk:create-pipeline-layout device layout-info))
