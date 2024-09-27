# PoCL Security Policy

PoCL developers are committed to the highest level of quality especially related
to security issues in the main branch that forms the basis for PoCL releases.
This security policy outlines the methods and steps taken to handle security issues.

## Types of Issues Considered Security Issues

OpenCL programs have primary goal of performance. Since PoCL is an OpenCL
implementation can invoke any OpenCL/SPIR-V-defined code in the user's 
devices. However, sometimes mitigating against such malicious programs with
software means leads to significant performance impact. Therefore, malicious
programs injected via user kernels are out of scope, unless the vulnerability
can be mitigated without any performance impact.

However, vulnerabilities such as buffer overflows or exploitable race conditions
present in the core runtime, which is implemented in C on the purpose of maximum
portability and lightness, are of high interest.

## Security Patch Supported Versions

The latest release version is supported with security patches, if any are
required, until the next major release. This policy started from release 6.0.

| Version | Supported          |
| ------- | ------------------ |
| 6.0     | :white_check_mark: |

## Reporting a Vulnerability

To report any 0-day vulnerabilities you have found in the code base, please
send an email to all the lead maintainers of the project:
pekka.jaaskelainen@intel.com, michal.babej@intel.com and jan.solanti@tuni.fi.

Another option to report the vulnerability is to utilize the Github form
https://github.com/pocl/pocl/security/advisories/new

After your submission, someone from the team will contact you within 7 days
with information on how your disclosure was or will be mitigated and when the
fix will be available in a security update release.
