# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import classification_pb2 as classification__pb2


class ClassImgServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ClassImg = channel.unary_unary(
                '/ClassImgService/ClassImg',
                request_serializer=classification__pb2.RequestClassImg.SerializeToString,
                response_deserializer=classification__pb2.ResponseClassImg.FromString,
                )


class ClassImgServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ClassImg(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ClassImgServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ClassImg': grpc.unary_unary_rpc_method_handler(
                    servicer.ClassImg,
                    request_deserializer=classification__pb2.RequestClassImg.FromString,
                    response_serializer=classification__pb2.ResponseClassImg.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'ClassImgService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ClassImgService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ClassImg(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ClassImgService/ClassImg',
            classification__pb2.RequestClassImg.SerializeToString,
            classification__pb2.ResponseClassImg.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)